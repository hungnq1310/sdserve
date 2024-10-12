import argparse
import os
import shutil
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from packaging import version

from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline

from sdserve.models.unet.cnet import UNet2DConditionXLControlNetModel

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
is_torch_2_0_1 = version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.1")

    
class StableDiffusionXLConverter:
    def __init__(
        self, model_path: str, output_path: str, opset: int, fp16: bool = False
    ):
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.opset = opset
        self.fp16 = fp16
        self.device = "cpu"
        self.dtype = torch.float16 if self.fp16 else torch.float32

    @torch.no_grad()
    def convert_models(
        self,
    ):
        """
        Function to convert models in stable diffusion controlnet pipeline into ONNX format.

        Returns:
            create 4 onnx models in output path
            text_encoder/model.onnx
            unet/model.onnx + unet/weights.pb
            vae_encoder/model.onnx
            vae_decoder/model.onnx]
        """
    
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_path, torch_dtype=self.dtype
        ).to(self.device)

        if is_torch_2_0_1:
            pipeline.unet.set_attn_processor(AttnProcessor())
            pipeline.vae.set_attn_processor(AttnProcessor())

        # # TEXT ENCODER
        num_tokens, text_hidden_size = self._convert_text_encoder(pipeline)
    
        # # UNET
        self._convert_unet(pipeline, num_tokens, text_hidden_size)

        # VAE ENCODER
        self._convert_vae(pipeline)

        del pipeline

    def _convert_text_encoder(self, pipeline):
        # # TEXT ENCODER
        num_tokens = pipeline.text_encoder.config.max_position_embeddings
        text_hidden_size = pipeline.text_encoder.config.hidden_size
        text_input = pipeline.tokenizer(
            "A sample prompt",
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        onnx_export(
            pipeline.text_encoder,
            # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
            model_args=(text_input.input_ids.to(device=self.device, dtype=self.torch.int32)),
            output_path=self.output_path / "text_encoder" / "model.onnx",
            ordered_input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
            },
            opset=self.self.opset,
        )
        del pipeline.text_encoder
        return num_tokens, text_hidden_size

    def _convert_unet(self, pipeline, num_tokens):
        # # UNET
        #! FIX THIS
        controlnets = torch.nn.ModuleList(controlnets)
        unet_controlnet = UNet2DConditionXLControlNetModel(pipeline.unet, controlnets)
        unet_in_channels = pipeline.unet.config.in_channels
        unet_sample_size = pipeline.unet.config.sample_size
        #? WHY 2048
        text_hidden_size = 2048
        img_size = 8 * unet_sample_size
        unet_path = self.output_path / "unet" / "model.onnx" 

        model_args = (
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=self.device, dtype=self.dtype),
            torch.tensor([1.0]).to(device=self.device, dtype=self.dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=self.device, dtype=self.dtype),
            torch.randn(len(controlnets), 2, 3, img_size, img_size).to(device=self.device, dtype=self.dtype),
            torch.randn(len(controlnets), 1).to(device=self.device, dtype=self.dtype),
            torch.randn(2, 1280).to(device=self.device, dtype=self.dtype),
            torch.rand(2, 6).to(device=self.device, dtype=self.dtype),
        )
        onnx_export(
            unet_controlnet,
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
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "2B"},
                "controlnet_conds": {1: "2B", 3: "8H", 4: "8W"},
                "text_embeds": {0: "2B"},
                "time_ids": {0: "2B"},
            },
            opset=self.opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        # optimize onnx
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = optimize(onnx.load(unet_model_path), name="Unet", verbose=True)
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
        del pipeline.unet
        return unet_sample_size

    def _convert_vae(self, pipeline, unet_sample_size):
        # VAE ENCODER
        vae_encoder = pipeline.vae
        vae_in_channels = vae_encoder.config.in_channels
        vae_sample_size = vae_encoder.config.sample_size
        # need to get the raw tensor output (sample) from the encoder
        vae_encoder.forward = lambda sample: vae_encoder.encode(sample).latent_dist.sample()
        onnx_export(
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
        vae_decoder = pipeline.vae
        vae_latent_channels = vae_decoder.config.latent_channels
        # forward only through the decoder part
        vae_decoder.forward = vae_encoder.decode
        onnx_export(
            vae_decoder,
            model_args=(
                torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=self.device, dtype=self.dtype),
            ),
            output_path=self.output_path / "vae_decoder" / "model.onnx",
            ordered_input_names=["latent_sample"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=self.opset,
        )
        del pipeline.vae