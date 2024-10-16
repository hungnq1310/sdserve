import torch
from transformers import CLIPTextModel

class CLIPTextWrapper(torch.nn.Module):
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name_or_path, **kwargs
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        result = self.text_encoder(
            input_ids=input_ids,
            return_dict=False
        )
        return result[0] # return the last hidden state