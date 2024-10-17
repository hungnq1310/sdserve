import torch
from typing import Optional, Tuple, Union
from diffusers import UNet2DConditionModel

class UnetWrapper(torch.nn.Module):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()

        self.unet: UNet2DConditionModel = model
        self.is_cnext = kwargs.get("is_cnext", False)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        add_image_embeds: Optional[torch.Tensor] = None,
        add_text_embeds: Optional[torch.Tensor] = None,
        add_time_ids: Optional[torch.Tensor] = None,
        add_hint: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple:
        
        #! ONLY cross_attention_kwargs is remove because undefined clearly use-case
        controls = {}
        if mid_block_additional_residual and scale:
            controls = {
                "out": mid_block_additional_residual,
                "scale": scale
            }

        if add_text_embeds and add_time_ids:
            added_cond_kwargs = {
                "time_ids": add_time_ids,
                "text_embeds": add_text_embeds
            }
        elif add_image_embeds and add_text_embeds:
            added_cond_kwargs = {
                "image_embeds": add_time_ids,
                "text_embeds": add_text_embeds
            }
        elif add_hint and add_image_embeds:
            added_cond_kwargs = {
                "image_embeds": add_time_ids,
                "hint": add_hint
            }
        elif add_image_embeds:
            added_cond_kwargs = {
                "image_embeds": add_time_ids,
            }
        
        if self.is_cnext:
            sample = self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                controls=controls,
                return_dict=False
            )
        else:
            sample = self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                controls=controls,
                return_dict=False
            )
        return sample