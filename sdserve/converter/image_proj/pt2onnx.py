import torch
from pathlib import Path

from sdserve.models.image_proj.wrapper import CLIPVisionWrapper
from sdserve.converter.onnx_v2 import OnnxConverter

class ImageEncoderConverter(OnnxConverter):
    def __init__(self,
        model_path: str, output_path: str, opset: int, fp16: bool   
    ) -> None:
        # load from any huggingface model 
        self.image_encoder = CLIPVisionWrapper(model_path=model_path, subfolder='models/image_encoder')

        # quantize the model
        self.dtype=torch.float32 if not fp16 else self.dtype
        self.device = "cpu"
        # set output path
        output_path = Path(output_path)
        self.cnet_path = output_path / "image_encoder" / "model.onnx"
        # setstuff
        self.opset = opset
        self.fp16 = fp16

    @torch.no_grad()
    def convert(self):
        """
        Convert the controlnet model to ONNX format

        Args:
            model_path (str): path to the model
            output_path (str): path to save the ONNX model
            opset (int): ONNX opset version
            fp16 (bool): convert the model to FP16
            attention_slicing (str): When "auto", input to the attention heads is halved, so attention is computed in two steps. If "max", maximum amount of memory is saved by running only one slice at a time. If a number is provided, uses as many slices as attention_head_dim // slice_size. In this case, attention_head_dim must be a multiple of slice_size. When this option is enabled, the attention module splits the input tensor in slices to compute attention in several steps. This is useful for saving some memory in exchange for a small decrease in speed.
        """

        # get the signature of the forward method
        inputs_args = self.get_inputs_names(self.image_encoder.forward)

        dump_inputs = (
            torch.randn(2, 3, 224, 224).to(device=self.device, dtype=self.dtype),
        )
        output_names = ["embedding"]
        dynamic_axes={
            'image_proc': {0: 'batch',1: 'channel', 2: 'height', 3:'weidth'},
            'embedding': {0:'batch', 1: 'embedding'} 
        }

        print("Exporting controlnet to ONNX...")
        self.onnx_export(
            self.image_encoder,
            model_args=dump_inputs,
            output_path=self.cnet_path,
            ordered_input_names=inputs_args,
            output_names=output_names,  #* has to be different from "sample" for correct tracing
            dynamic_axes=dynamic_axes, #* timestep and condition_scale are not dynamic axes
            opset=self.opset,
        )
        print("Controlnet exported to ONNX successfully at: " + str(self.cnet_path.absolute().as_posix()))