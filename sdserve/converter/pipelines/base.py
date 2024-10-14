import os
import shutil

import onnx
import torch
from onnx import shape_inference
from packaging import version

from diffusers import (
    AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
)
from diffusers.models.attention_processor import AttnProcessor

from sdserve.converter.onnx_v2 import OnnxConverter

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
is_torch_2_0_1 = version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.1")


class StableDiffusionConverter(OnnxConverter):
    """
    This class is used to convert the stable diffusion model to ONNX format.
    All functionalities of `diffusers` are preserved. 
    Refer - https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview#load-community-pipelines-and-components

    Functions:
    - Convert any size of stable diffusion model to ONNX format. (focus only on task-specific)
    - Convert any customization from community:
        - controlnet
        - unet
        - text encoder
        - image processor
        - vae 
    """
    def __init__(
        self,
        model_path: str,
        task: str,
        output_path: str,
        opset: int,
        fp16: bool = False,
        xformers: bool = False,
        vae_slicing: bool = False,
        vae_tiling: bool = False,
        cpu_offload: bool = False,
        **kwargs,
        ) -> None:
        super().__init__()

        # Initialize the model
        assert task in ["text-to-image", "image-to-image", "inpainting"], "Invalid task. Choose from 'text-to-image', 'image-to-image', 'inpainting'."
        if task == "text-to-image":
            self.model = AutoPipelineForText2Image.from_pretrained(model_path, **kwargs) 
        elif task == "image-to-image":
            self.model = AutoPipelineForImage2Image.from_pretrained(model_path, **kwargs)
        elif task == "inpainting":
            self.model = AutoPipelineForInpainting.from_pretrained(model_path, **kwargs)

        # Memory efficiency
        """`difussers` v0.30.0"""
        if vae_slicing:
            """
            a small performance boost in VAE decoding on multi-image batches, and there should be no performance impact on single-image batches.
            want to couple this with enable_xformers_memory_efficient_attention() to reduce more memory usage.
            """
            self.model.vae_slicing = True
        if vae_tiling:
            """
            working with large images on limited VRAM (for example, generating 4k images on 8GB of VRAM).
            should also used tiled VAE with pipe.enable_xformers_memory_efficient_attention() to reduce more memory usage.
            """
        if cpu_offload:
            """
            When using enable_sequential_cpu_offload(), don't move the pipeline to CUDA beforehand or else the gain in memory consumption will only be minimal (https://github.com/huggingface/diffusers/issues/1934).
            Consider using model offloading if you want to optimize for speed because it is much faster. The tradeoff is your memory savings won't be as large.
            """
        if xformers:
            """
            This method is used to enable memory-efficient attention for XFormers. If you have PyTorch >= 2.0 installed, you should not expect a speed-up for inference when enabling xformers.
            """
            self.model.enable_xformers_memory_efficient_attention()

        # Attention Processor
        if is_torch_2_0_1:
            self.model.unet.set_attn_processor(AttnProcessor())
            self.model.vae.set_attn_processor(AttnProcessor())
        
        # setstuff
        self.opset = opset
        self.fp16 = fp16
        self.task = task
        self.output_path = output_path
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.device = "cpu"


    @torch.no_grad()
    def convert(self):
        """
        Function to convert models in stable diffusion controlnet pipeline into ONNX format.

        Returns:
            create 4 onnx models in output path
            text_encoder/model.onnx
            unet/model.onnx + unet/weights.pb
            vae_encoder/model.onnx
            vae_decoder/model.onnx]
        """
        # # TEXT ENCODER
        self._convert_text_encoder()
        
        # # UNET
        self._convert_unet()

        # VAE ENCODER
        self._convert_vae()

        del pipeline

    def _convert_text_encoder(self):
        # # TEXT ENCODER
        text_input = self.model.tokenizer(
            "A sample prompt",
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        #! cast int32 here
        model_inputs = (text_input.input_ids.to(device=self.device, dtype=torch.int32))
        self.onnx_export(
            self.model.text_encoder,
            # casting to torch.int32 until the CLIP fix is released: 
            # https://github.com/huggingface/transformers/pull/18515/files
            model_args=model_inputs,
            output_path=self.output_path / "text_encoder" / "model.onnx",
            ordered_input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
            },
            opset=self.opset,
        )
        del self.model.text_encoder

    def _convert_unet(self):
        # text encoder shapes
        num_tokens = self.model.text_encoder.config.max_position_embeddings
        text_hidden_size = self.model.text_encoder.config.hidden_size
        # unet shapes
        unet_in_channels = self.model.unet.config.in_channels
        unet_sample_size = self.model.unet.config.sample_size
        img_size = 8 * unet_sample_size
        unet_path = self.output_path / "unet" / "model.onnx"

        self.onnx_export(
            self.model.unet,
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=self.device, dtype=self.dtype),
                torch.tensor([1.0]).to(device=self.device, dtype=self.dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(device=self.device, dtype=self.dtype),
                torch.randn(2, 3, img_size, img_size).to(device=self.device, dtype=self.dtype),
                torch.randn(2).to(device=self.device, dtype=self.dtype),
            ),
            output_path=unet_path,
            ordered_input_names=[
                "sample",
                "timestep",
                "encoder_hidden_states",
                "controlnet_conds",
                "conditioning_scales",
            ],
            output_names=["noise_pred"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "controlnet_conds": {0: "batch", 3: "height", 4: "width"},
                "conditioning_scales": {0: "batch"},
            },
            opset=self.opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        # optimize onnx
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = self.optimize(onnx.load(unet_model_path), name="Unet", verbose=True)
        # clean up existing tensor files
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        # collate external tensor files into one
        onnx.save_model(
            unet_opt_graph,
            unet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        del self.model.unet
        

    def _convert_vae(self):
        # Unet 
        unet_sample_size = self.model.unet.config.sample_size
        # VAE ENCODER
        vae_encoder = self.model.vae
        vae_in_channels = vae_encoder.config.in_channels
        vae_sample_size = vae_encoder.config.sample_size
        # need to get the raw tensor output (sample) from the encoder
        vae_encoder.forward = lambda sample: vae_encoder.encode(sample).latent_dist.sample()
        self.onnx_export(
            vae_encoder,
            model_args=(torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=self.device, dtype=self.dtype),),
            output_path=self.output_path / "vae_encoder" / "model.onnx",
            ordered_input_names=["sample"],
            output_names=["latent_sample"],
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=self.opset,
        )

        # VAE DECODER
        vae_decoder = self.model.vae
        vae_latent_channels = vae_decoder.config.latent_channels
        # forward only through the decoder part
        vae_decoder.forward = vae_encoder.decode
        model_args =(
            torch.randn(
                1, vae_latent_channels, unet_sample_size, unet_sample_size
            ).to(device=self.device, dtype=self.dtype),
        )
        self.onnx_export(
            vae_decoder,
            model_args=model_args,
            output_path=self.output_path / "vae_decoder" / "model.onnx",
            ordered_input_names=["latent_sample"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=self.opset,
        )
        del self.model.vae

   
class StableDiffusionXLConverter(StableDiffusionConverter):

    def convert(self):
        #TODO: Extend more components: text_encoder_2, image_processor, vae_2
        pass
    
    def _convert_unet(self):
        num_tokens = self.model.text_encoder.config.max_position_embeddings
        # # UNET
        unet_in_channels = self.model.unet.config.in_channels
        unet_sample_size = self.model.unet.config.sample_size
        #? WHY 2048
        text_hidden_size = 2048
        img_size = 8 * unet_sample_size
        unet_path = self.output_path / "unet" / "model.onnx" 

        model_args = (
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=self.device, dtype=self.dtype),
            torch.tensor([1.0]).to(device=self.device, dtype=self.dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=self.device, dtype=self.dtype),
            torch.randn(2, 3, img_size, img_size).to(device=self.device, dtype=self.dtype),
            torch.randn(2).to(device=self.device, dtype=self.dtype),
            torch.randn(2, 1280).to(device=self.device, dtype=self.dtype),
            torch.rand(2, 6).to(device=self.device, dtype=self.dtype),
        )
        self.onnx_export(
            self.model.unet,
            model_args=model_args,
            output_path=unet_path,
            ordered_input_names=[
                "sample",
                "timestep",
                "encoder_hidden_states",
                "controlnet_conds",
                "conditioning_scales",
                "text_embeds",
                "time_ids",
            ],
            output_names=["noise_pred"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "controlnet_conds": {0: "batch", 3: "height", 4: "width"},
                "conditioning_scales": {0: "batch"},
                "text_embeds": {0: "batch"},
                "time_ids": {0: "batch"},
            },
            opset=self.opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        # optimize onnx
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = self.optimize(onnx.load(unet_model_path), name="Unet", verbose=True)
        # clean up existing tensor files
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        # collate external tensor files into one
        onnx.save_model(
            unet_opt_graph,
            unet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        del self.model.unet
        return unet_sample_size