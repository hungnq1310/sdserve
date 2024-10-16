from safetensors.torch import load_file
import torch
from sdserve.models.cnext.wrapper import ControlNeXt, ControlNeXtXL
from sdserve.converter.onnx_v2 import OnnxConverter

def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        state_dict = load_file(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

#TODO: convert this to class and inherit from OnnxConverter
class ControlNeXtConverter(OnnxConverter):
    def __init__(self, 
                 config: dict, 
                 ckpt_path: str, 
                 output_path: str, 
                 opset_version: int = 14, 
                 do_constant_folding: bool = True,
                 is_sdxl: bool = False
                ):
        # load from checkpoint
        if is_sdxl:
            self.controlnext = ControlNeXtXL.from_config(config)
        else:
            self.controlnext = ControlNeXt.from_config(config)
        load_safetensors(self.controlnext, ckpt_path)
        self.output_path = output_path
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding

    @torch.no_grad()
    def convert(self):
        """
        Convert the controlnet model to ONNX format
        """
        # dummy inputs
        image = torch.randn((1, 3, 512, 512), dtype=torch.float32)
        timestep = torch.randint(low=0, high=10, size=(1, ), dtype=torch.int32)
        dummy_inputs = (image, timestep)
        # export to onnx
        onnx_output_path = self.output_path + "/model.onnx"
        torch.onnx.export(
            self.controlnext,
            dummy_inputs,               
            onnx_output_path,              
            opset_version=self.opset_version,           
            do_constant_folding=self.do_constant_folding,   
            input_names=['image', 'timestep'],  
            output_names=['sample', 'cnext-scale'],    
            dynamic_axes={
                'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'timestep': {0: 'batch_size'},
                'sample': {0: 'batch_size'},
            }
        )