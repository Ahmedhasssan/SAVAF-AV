import math
import einops
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import *
from gaussian_feature_map import *
import time


def filter_gaussians_by_position(
    xyz, 
    center=None, 
    radius=None, 
    bounds=None,
    camera_pos=None,
    view_dir=None,
    view_angle=None,
    max_distance=None
):
    N = xyz.shape[0]
    device = xyz.device
    mask = torch.ones(N, dtype=torch.bool, device=device)
    if center is not None and radius is not None:
        center = torch.tensor(center, dtype=xyz.dtype, device=device)
        distances = torch.norm(xyz - center.unsqueeze(0), dim=1)
        mask &= (distances <= radius)
    if bounds is not None:
        min_xyz, max_xyz = bounds
        min_xyz = torch.tensor(min_xyz, dtype=xyz.dtype, device=device)
        max_xyz = torch.tensor(max_xyz, dtype=xyz.dtype, device=device)
        within_bounds = (xyz >= min_xyz.unsqueeze(0)) & (xyz <= max_xyz.unsqueeze(0))
        mask &= torch.all(within_bounds, dim=1)
    if camera_pos is not None:
        camera_pos = torch.tensor(camera_pos, dtype=xyz.dtype, device=device)
        to_gaussian = xyz - camera_pos.unsqueeze(0)
        distances = torch.norm(to_gaussian, dim=1)
        if max_distance is not None:
            mask &= (distances <= max_distance)
        if view_dir is not None and view_angle is not None:
            view_dir = torch.tensor(view_dir, dtype=xyz.dtype, device=device)
            view_dir = view_dir / torch.norm(view_dir)
            to_gaussian_normalized = to_gaussian / distances.unsqueeze(1)
            cos_angles = torch.sum(to_gaussian_normalized * view_dir.unsqueeze(0), dim=1)
            cos_threshold = np.cos(np.radians(view_angle))
            mask &= (cos_angles >= cos_threshold)
    indices = torch.nonzero(mask).squeeze(1)
    return indices


class Embedding(nn.Module):
    def __init__(self, num_layer, num_embed, ch):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(num_embed, num_layer, ch) / math.sqrt(ch), requires_grad=True)
        self.num_embed = num_embed

    def forward(self, ori):
        ori = ori.cuda()
        embeds = torch.cat([self.embeds[-1:], self.embeds, self.embeds[:1]], dim=0)
        ori = (ori + 1) / 2 * self.num_embed
        t_value = torch.arange(-1, self.num_embed + 1, device=ori.device)
        right_idx = torch.searchsorted(t_value, ori, right=False)
        left_idx = right_idx - 1

        left_dis = ori - t_value[left_idx]
        right_dis = t_value[right_idx] - ori
        left_dis = torch.clamp(left_dis, 0, 1).unsqueeze(1).unsqueeze(2)
        right_dis = torch.clamp(right_dis, 0, 1).unsqueeze(1).unsqueeze(2)

        left_embed = embeds[left_idx]
        right_embed = embeds[right_idx]

        output = left_embed * right_dis + right_embed * left_dis
        return output


class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=-1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0, stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        x_input = x_input.cuda()
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input * freq))
        return torch.cat(out_list, dim=self.ch_dim)


def mydropout(tensor, p=0.5, training=True):
    if not training or p == 0:
        return tensor
    else:
        batch_size = tensor.shape[0]
        random_tensor = torch.rand(batch_size, device=tensor.device)
        new_tensor = [torch.zeros_like(tensor[i]) if random_tensor[i] <= p else tensor[i] for i in range(batch_size)]
        new_tensor = torch.stack(new_tensor, dim=0)
        return new_tensor


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, context):
        B, N, _, H = *query.shape, self.heads
        _, M, _ = context.shape
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = einops.rearrange(q, 'b n (h d) -> b h n d', h=H)
        k = einops.rearrange(k, 'b m (h d) -> b h m d', h=H)
        v = einops.rearrange(v, 'b m (h d) -> b h m d', h=H)
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query):
        B, N, _, H = *query.shape, self.heads
        q, k, v = self.to_qkv(query).chunk(3, dim=-1)
        q = einops.rearrange(q, 'b n (h d) -> b h n d', h=H)
        k = einops.rearrange(k, 'b m (h d) -> b h m d', h=H)
        v = einops.rearrange(v, 'b m (h d) -> b h m d', h=H)
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MixDiffWithCrossAttention(nn.Module):
    def __init__(self, conv, resolution=(64, 180), query_dim=128, intermediate_dim=128, freq_num=257, ori_dim=42, heads=4, dim_head=8, p=0, training=True):
        super().__init__()
        # Resolution of the projected Gaussian feature map. Lower than the original
        # (170, 480) to keep the flat projection layer compact; aspect ratio is preserved.
        self.resolution = tuple(resolution)
        self.feature_dim = self.resolution[0] * self.resolution[1]
        self.query_proj = nn.Sequential(nn.Linear(42 + 21, intermediate_dim), nn.ReLU(inplace=True))
        self.feature_proj = nn.Linear(self.feature_dim, intermediate_dim).cuda()
        self.cross_attention_mix = SelfAttention(intermediate_dim, heads, dim_head).cuda()
        self.cross_attention_diff = CrossAttention(intermediate_dim, heads, dim_head).cuda()
        self.ori_embedder = Embedding(4, 4, intermediate_dim)

        self.mix_prj = nn.Linear(intermediate_dim, 1).cuda()
        self.diff_prj = nn.Linear(intermediate_dim, 1).cuda()

        self.conv = conv
        self.freq_num = freq_num
        self.pos_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1)

        if self.conv:
            self.post_process = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 3, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 3, 1, 1)).cuda()

        self.p = p
        self.training = training

    def forward(self, x, gaussians):
        B = x.pos.shape[0]
        pos = self.pos_embedder(x.pos)
        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x.pos.device).unsqueeze(1)
        freq = self.freq_embedder(freq)

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos.float(), freq], dim=2)
        query = self.query_proj(query).cuda()

        camera_matrix = torch.matmul(x.full_proj_transform, x.world_view_transform)
        view_matrix = x.world_view_transform
        camera_pos = -torch.matmul(view_matrix[:3, :3].transpose(0, 1), view_matrix[:3, 3])
        selected_indices = filter_gaussians_by_position(xyz=gaussians.get_xyz, camera_pos=camera_pos, max_distance=0.5)

        filtered_xyz = gaussians.get_xyz[selected_indices]
        filtered_scaling = gaussians.get_scaling[selected_indices]
        filtered_rotation = gaussians.get_rotation[selected_indices]
        filtered_opacity = gaussians.get_opacity[selected_indices]

        audio_feats = create_gaussian_feature_map(
            filtered_xyz,
            filtered_scaling,
            filtered_rotation,
            filtered_opacity,
            camera_matrix,
            resolution=self.resolution,
            feature_dim=1,
        )
        v_feats = audio_feats["feature_map"]
        v_feats = v_feats.reshape(1, self.feature_dim)
        if self.training:
            noise = torch.randn_like(v_feats) * 0.1
            v_feats = (v_feats + noise).cuda()

        features = self.feature_proj(v_feats)
        features = mydropout(features, p=self.p, training=self.training)
        features = einops.repeat(features, "b c -> b 1 c")

        mix_features = self.cross_attention_mix(query + features)
        mask_mix = self.mix_prj(mix_features).squeeze(-1)

        ori = self.ori_embedder(x.ori)
        ori = mydropout(ori, p=self.p, training=self.training).float().cuda()
        diff_features = self.cross_attention_diff(mix_features, ori)
        mask_diff = self.diff_prj(diff_features).squeeze(-1)
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1

        x.mag_sc = x.mag_sc.cuda()

        time_dim = x.mag_sc.shape[1]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim)
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim)
        reconstr_mono = x.mag_sc * mask_mix
        reconstr_diff = reconstr_mono * mask_diff
        reconstr_left = reconstr_mono + reconstr_diff
        reconstr_right = reconstr_mono - reconstr_diff

        if self.conv:
            left_input = torch.stack([x.mag_sc, mask_mix, mask_diff, reconstr_left], dim=1)
            right_input = torch.stack([x.mag_sc, mask_mix, -mask_diff, reconstr_right], dim=1)
            left_input = left_input.float()
            right_input = right_input.float()
            left_output = self.post_process(left_input).squeeze(1)
            right_output = self.post_process(right_input).squeeze(1)
            reconstr_left = reconstr_left + left_output
            reconstr_right = reconstr_right + right_output
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1)
            reconstr = F.relu(reconstr)
        else:
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1)
            reconstr = F.relu(reconstr)
        return {"mask_mix": mask_mix,
                "mask_diff": mask_diff,
                "reconstr_mono": reconstr_mono,
                "reconstr": reconstr}
