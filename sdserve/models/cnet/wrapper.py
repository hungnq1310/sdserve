import torch
from diffusers import ControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers

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
    def __init__(self, controlnet: ControlNetModel, scheduler: KarrasDiffusionSchedulers):
        super().__init__()
        self.controlnet = controlnet
        self.scheduler = scheduler

    def forward(
        self, 
        latent_sample: torch.Tensor,        # positional argument
        timestep: torch.Tensor,             # positional argument
        prompt_embeds: torch.Tensor,        # positional argument
        cond_image: torch.Tensor,           # positional argument
        cond_scale: torch.Tensor,           # positional argument
        add_text_embeds: torch.Tensor,      # positional argument
        add_time_ids: torch.Tensor,         # positional argument
        controlnet_keep_i: list,            # positional argument
        guess_mode: bool,                   # keyword argument
        do_classifier_free_guidance: bool,  # keyword argument
        is_sdxl: bool,                      # keyword argument
        ):
        # controlnet(s) inference
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            latent_sample = self.scheduler.scale_model_input(latent_sample, timestep)
            prompt_embeds = prompt_embeds.chunk(2)[1]
            added_cond_kwargs["text_embeds"] = add_text_embeds.chunk(2)[1]
            added_cond_kwargs["time_ids"] = add_time_ids.chunk(2)[1]

        if isinstance(controlnet_keep_i, list):
            cond_scale = [c * s for c, s in zip(controlnet_cond_scale, controlnet_keep_i)]
        else:
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep_i

        if is_sdxl:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_sample,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        else:
            # not pass added_cond_kwargs
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_sample,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

        return down_block_res_samples, mid_block_res_sample
