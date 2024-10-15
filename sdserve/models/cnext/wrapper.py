# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class ControlProject(nn.Module):
    def __init__(self, num_channels, scale=8, is_empty=False) -> None:
        super().__init__()
        assert scale and scale & (scale - 1) == 0
        self.is_empty = is_empty
        self.scale = scale
        if not is_empty:
            if scale > 1:
                self.down_scale = nn.AvgPool2d(scale, scale)
            else:
                self.down_scale = nn.Identity()
            self.out = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=False)
            for p in self.out.parameters():
                nn.init.zeros_(p)

    def forward(
            self,
            hidden_states: torch.FloatTensor):
        if self.is_empty:
            shape = list(hidden_states.shape)
            shape[-2] = shape[-2] // self.scale
            shape[-1] = shape[-1] // self.scale
            return torch.zeros(shape).to(hidden_states)

        if len(hidden_states.shape) == 5:
            B, F, C, H, W = hidden_states.shape
            hidden_states = rearrange(hidden_states, "B F C H W -> (B F) C H W")
            hidden_states = self.down_scale(hidden_states)
            hidden_states = self.out(hidden_states)
            hidden_states = rearrange(hidden_states, "(B F) C H W -> B F C H W", F=F)
        else:
            hidden_states = self.down_scale(hidden_states)
            hidden_states = self.out(hidden_states)
        return hidden_states


class ControlNeXt(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: List[int] = [3, 128, 256],
        out_channels: List[int] = [128, 256, 256],
        groups: List[int] = [1, 8, 8],
        time_embed_dim: int = 256,
        final_out_channels: int = 1280,
    ):
        super().__init__()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)

        self.down_res = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(len(in_channels)):
            self.down_res.append(
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i]
                ),
            )
            self.down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name="op",
                )
            )
        
        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
        ))
        self.mid_convs.append(nn.Conv2d(
            in_channels=out_channels[-1],
            out_channels=final_out_channels,
            kernel_size=1,
            stride=1,
        ))

        self.scale = nn.Parameter(torch.tensor(1.))


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> Tuple:
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb_batch = self.time_embedding(t_emb)

        emb = emb_batch

        for res, downsample in zip(self.down_res, self.down_sample):
            sample = res(sample, emb)
            sample = downsample(sample, emb)
        
        for mid_conv in self.mid_convs:
            sample = mid_conv(sample)
        
        return sample, self.scale
    

class ControlNeXtXL(ControlNeXt):

    @register_to_config
    def __init__(
        self,
        in_channels: List[int] = [128, 128],
        out_channels: List[int] = [128, 256],
        groups: List[int] = [4, 8],
        time_embed_dim: int = 256,
        final_out_channels: int = 320,
    ):
        super().__init__(in_channels, out_channels, groups, time_embed_dim, final_out_channels)

        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
        )
        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels[-1]),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GroupNorm(8, out_channels[-1]),
        ))
        self.mid_convs.append(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=final_out_channels,
                kernel_size=1,
                stride=1,
            ))

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> Tuple:

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)
        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        emb_batch = self.time_embedding(t_emb)

        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb_batch
        sample = self.embedding(sample)
        for res, downsample in zip(self.down_res, self.down_sample):
            sample = res(sample, emb)
            sample = downsample(sample, emb)
        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)
        return sample, self.scale