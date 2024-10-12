from safetensors.torch import load_file
import torch
from sdserve.models.cnext.cnext import ControlNeXt

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

def convert(ckpt_path: str, output_path: str, opset_version: int = 14, do_constant_folding: bool = True):
    """
    Convert the controlnet model to ONNX format
    """
    # load from checkpoint
    controlnet = ControlNeXt()
    load_safetensors(controlnet, ckpt_path)
    # dummy inputs
    image = torch.randn((1, 3, 512, 512), dtype=torch.float32)
    timestep = torch.randint(low=0, high=10, size=(1, ), dtype=torch.int32)
    dummy_inputs = (image, timestep)
    # export to onnx
    onnx_output_path = output_path + "/model.onnx"
    torch.onnx.export(
        controlnet,
        dummy_inputs,               
        onnx_output_path,              
        opset_version=opset_version,           
        do_constant_folding=do_constant_folding,   
        input_names=['image', 'timestep'],  
        output_names=['sample', 'cnet-scale'],    
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'timestep': {0: 'batch_size'},
            'sample': {0: 'batch_size'},
        }
    )