import torch
from typing import Optional
from diffusers import ControlNetModel

class ControlNetWrapper(torch.nn.Module):
    """
    ControlNetWrapper is a wrapper around ControlNetModel that allows flattening inputs and outputs
    of diffusers's ControlNetModel for ONNX conversion.
    
    Features:
    - Flattens the inputs and outputs of ControlNetModel
    - Supports both SDXL and SD15 ControlNetModel

    Code detached from StableDiffusionPipeline:
    - sdxl: https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L1456 (line L1456 to 1486)
    - sd15: https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#1246 (line 1246 to 1272)
    """
    def __init__(self, model_name_or_path: str, attention_slicing: Optional[str] = None) -> None:
        super().__init__()
        self.controlnet = ControlNetModel.from_pretrained(model_name_or_path)
        if attention_slicing is not None:
            self.controlnet.set_attention_slice(attention_slicing)

    def forward(
        self, 
        latent_sample: torch.Tensor,        # positional argument
        timestep: torch.Tensor,             # positional argument
        prompt_embeds: torch.Tensor,        # positional argument
        cond_image: torch.Tensor,           # positional argument
        cond_scale: torch.Tensor,           # positional argument
        add_text_embeds: Optional[torch.Tensor] = None,      # use for xl
        add_time_ids: Optional[torch.Tensor] = None,         # use for xl
        ):
        # controlnet(s) inference
        added_cond_kwargs = {}
        if add_text_embeds is not None:
            added_cond_kwargs["text_embeds"] = add_text_embeds  
        if add_time_ids is not None:
            added_cond_kwargs["time_ids"] = add_time_ids

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_sample,
            timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=cond_image,
            conditioning_scale=cond_scale,
            added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
            return_dict=False,
        )
        return tuple(down_block_res_samples), mid_block_res_sample # return tuple
