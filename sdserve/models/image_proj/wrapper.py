import torch
from transformers import CLIPVisionModelWithProjection

class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__()
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(model_name_or_path, **kwargs)


    def forward(self, image_proc: torch.Tensor):
        result = self.clip_model(image_proc, return_dict=False)
        return result[0]