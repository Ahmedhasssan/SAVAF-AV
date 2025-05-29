import math
import einops
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import *
from gaussian_feature_map import *
import time

class ANeRF(nn.Module):
    def __init__(self,
                 gaussians,
                 conv=False,
                 freq_num=257,
                 time_num=173,
                 intermediate_ch=128,
                 p=0):
        super(ANeRF, self).__init__()
        self.gaussians = gaussians
        self.conv = conv
        self.freq_num = freq_num
        self.pos_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.query_prj = nn.Sequential(nn.Linear(42 + 21, intermediate_ch), nn.ReLU(inplace=True))
        self.mix_mlp = MLPwSkip(intermediate_ch, intermediate_ch)
        self.mix_prj = nn.Linear(intermediate_ch, 1)
        self.ori_embedder = Embedding(4, 4, intermediate_ch)
        self.diff_mlp = MLPwSkip(intermediate_ch, intermediate_ch)
        self.diff_prj = nn.Linear(intermediate_ch, 1)
        self.av_mlp = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, intermediate_ch),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(intermediate_ch, intermediate_ch))
        if self.conv:
            self.post_process = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 3, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 3, 1, 1))
        
        self.p = p
        
    def forward(self, x):
        # x {"pos", "ori", "depth", "rgb"}
        B = x["pos"].shape[0]
        pos = self.pos_embedder(x["pos"]) # [B, 42]
        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x["pos"].device).unsqueeze(1) # [F, 1]
        freq = self.freq_embedder(freq) # [F, 21]

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos, freq], dim=2) # [B, F, ?]
        query = self.query_prj(query) # [B, F, ?]

        v_feats = torch.cat([x["rgb"], x["depth"]], dim=1) # [B, 1024]
        if self.training:
            noise = torch.randn_like(v_feats) * 0.1
            v_feats = v_feats + noise
        v_feats = self.av_mlp(v_feats) # [B, ?]
        v_feats = mydropout(v_feats, p=self.p, training=self.training)
        v_feats = einops.repeat(v_feats, "b c -> b 1 c")

        # predict mix mask
        feats = self.mix_mlp(query + v_feats)
        mask_mix = self.mix_prj(feats).squeeze(-1) # [B, F]

        # predict diff mask
        ori = self.ori_embedder(x["ori"])
        ori = mydropout(ori, p=self.p, training=self.training)
        feats = self.diff_mlp(feats, ori)
        mask_diff = self.diff_prj(feats).squeeze(-1) # [B, F]
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1

        time_dim = x["mag_sc"].shape[1]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim)
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim)
        reconstr_mono = x["mag_sc"] * mask_mix # [B, T, F]
        reconstr_diff = reconstr_mono * mask_diff # [B, T, F]
        reconstr_left = reconstr_mono + reconstr_diff
        reconstr_right = reconstr_mono - reconstr_diff
        
        if self.conv:
            left_input = torch.stack([x["mag_sc"], mask_mix, mask_diff, reconstr_left], dim=1) # [B, 4, T, F]
            right_input = torch.stack([x["mag_sc"], mask_mix, -mask_diff, reconstr_right], dim=1) # [B, 4, T, F]
            left_output = self.post_process(left_input).squeeze(1)
            right_output = self.post_process(right_input).squeeze(1)
            reconstr_left = reconstr_left + left_output
            reconstr_right = reconstr_right + right_output
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)
        else:
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)

        return {"mask_mix": mask_mix, 
                "mask_diff": mask_diff,
                "reconstr_mono": reconstr_mono,
                "reconstr": reconstr}


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
            view_dir = view_dir / torch.norm(view_dir)  # Normalize
            to_gaussian_normalized = to_gaussian / distances.unsqueeze(1)
            cos_angles = torch.sum(to_gaussian_normalized * view_dir.unsqueeze(0), dim=1)
            cos_threshold = np.cos(np.radians(view_angle))
            mask &= (cos_angles >= cos_threshold)
    indices = torch.nonzero(mask).squeeze(1)
    return indices

class ANeRF_V2(nn.Module):
    def __init__(self,
                 conv=False,
                 freq_num=257,
                 time_num=173,
                 intermediate_ch=128,
                 p=0):
        super(ANeRF_V2, self).__init__()
        self.conv = conv
        self.freq_num = freq_num
        self.pos_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.query_prj = nn.Sequential(nn.Linear(42 + 21, intermediate_ch), nn.ReLU(inplace=True))
        self.mix_mlp = MLPwSkip(intermediate_ch, intermediate_ch).cuda()
        self.mix_prj = nn.Linear(intermediate_ch, 1).cuda()
        self.ori_embedder = Embedding(4, 4, intermediate_ch)
        self.diff_mlp = MLPwSkip(intermediate_ch, intermediate_ch).cuda()
        self.diff_prj = nn.Linear(intermediate_ch, 1).cuda()
        self.max_in_channel = 5000000
        # self.av_mlp =nn.Sequential(nn.Linear(self.max_in_channel, 512), nn.Linear(512, intermediate_ch)).cuda()
        self.linear_feat =nn.Linear(self.max_in_channel, 512).cuda()
        self.av_mlp =nn.Linear(512, intermediate_ch).cuda()
        if self.conv:
            self.post_process = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 3, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 3, 1, 1)).cuda()
        
        self.p = p
        
    def forward(self, x, gaussians):
        # x {"pos", "ori", "depth", "rgb"}
        B = x.pos.shape[0]
        pos = self.pos_embedder(x.pos) # [B, 42]
        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x.pos.device).unsqueeze(1) # [F, 1]
        freq = self.freq_embedder(freq) # [F, 21]

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos.float(), freq], dim=2) # [B, F, ?]
        query = self.query_prj(query) # [B, F, ?]

        # v_feats = gaussians.get_features.flatten(0).unsqueeze(0)
        camera_matrix = torch.matmul(x.full_proj_transform, x.world_view_transform)
        view_matrix = x.world_view_transform
        camera_pos = -torch.matmul(view_matrix[:3, :3].transpose(0, 1), view_matrix[:3, 3])
        import pdb;pdb.set_trace()
        selected_indices = filter_gaussians_by_position(xyz=gaussians.get_xyz, camera_pos=camera_pos,  max_distance=0.8)

        filtered_xyz = gaussians.get_xyz[selected_indices]
        filtered_scaling = gaussians.get_scaling[selected_indices]
        filtered_rotation = gaussians.get_rotation[selected_indices]
        filtered_opacity = gaussians.get_opacity[selected_indices]

        audio_feats = create_gaussian_feature_map(
            filtered_xyz, 
            filtered_scaling, 
            filtered_rotation, 
            filtered_opacity,
            camera_matrix
        )
        # audio_feats = create_gaussian_feature_map(gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation, gaussians.get_opacity , camera_matrix)

        # audio_feats = compute_gaussian_audio_features(gaussians.get_xyz, gaussians.get_scale, gaussians.ge_rotation, gaussians.get_opacity, )

        # v_feats = torch.cat([x["rgb"], x["depth"]], dim=1) # [B, 1024]
        v_feats = audio_feats["feature_map"]
        v_feats = v_feats.view(v_feats.size(0), -1)
        if self.training:
            noise = torch.randn_like(v_feats) * 0.1
            v_feats = (v_feats + noise).cuda()
        # feats_linear = nn.Linear(gaussians.get_features.flatten(0).size(0),512).cuda()
        # v_feats_linear = feats_linear(v_feats.cuda())
            
        # assert gaussians.get_features.flatten(0).size(0) <= self.max_in_channel, "Requested in_channels exceeds max_in_channels"
        # v_feats = v_feats[:, :gaussians.get_features.flatten(0).size(0)]  # Slice input to desired size
        # weight = self.linear_feat.weight[:, :gaussians.get_features.flatten(0).size(0)]  # Slice weights
        # bias = self.linear_feat.bias
        # v_feats = nn.functional.linear(v_feats, weight, bias)

        v_feats = self.av_mlp(v_feats) # [B, ?]
        # import pdb;pdb.set_trace()
        v_feats = mydropout(v_feats, p=self.p, training=self.training)
        v_feats = einops.repeat(v_feats, "b c -> b 1 c")

        # predict mix mask
        feats = self.mix_mlp(query.cuda() + v_feats)
        mask_mix = self.mix_prj(feats).squeeze(-1) # [B, F]

        # predict diff mask
        # import pdb;pdb.set_trace()
        # ori = self.ori_embedder(x["ori"])
        ori = self.ori_embedder(x.ori)
        ori = mydropout(ori, p=self.p, training=self.training)
        feats = self.diff_mlp(feats, ori.float().cuda())
        mask_diff = self.diff_prj(feats).squeeze(-1) # [B, F]
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1

        # time_dim = x["mag_sc"].shape[1]
        time_dim = x.mag_sc.shape[1]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim)
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim)
        # reconstr_mono = x["mag_sc"] * mask_mix # [B, T, F]
        reconstr_mono = x.mag_sc.cuda() * mask_mix # [B, T, F]
        reconstr_diff = reconstr_mono * mask_diff # [B, T, F]
        reconstr_left = reconstr_mono + reconstr_diff
        reconstr_right = reconstr_mono - reconstr_diff
        
        if self.conv:
            left_input = torch.stack([x["mag_sc"], mask_mix, mask_diff, reconstr_left], dim=1) # [B, 4, T, F]
            right_input = torch.stack([x["mag_sc"], mask_mix, -mask_diff, reconstr_right], dim=1) # [B, 4, T, F]
            left_output = self.post_process(left_input).squeeze(1)
            right_output = self.post_process(right_input).squeeze(1)
            reconstr_left = reconstr_left + left_output
            reconstr_right = reconstr_right + right_output
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)
        else:
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)

        return {"mask_mix": mask_mix, 
                "mask_diff": mask_diff,
                "reconstr_mono": reconstr_mono,
                "reconstr": reconstr}

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
        left_dis = torch.clamp(left_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        right_dis = torch.clamp(right_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]

        left_embed = embeds[left_idx] # [B, l, c]
        right_embed = embeds[right_idx] # [B, l, c]

        output = left_embed * right_dis + right_embed * left_dis
        return output # [B, l, c]


class MLPwSkip(nn.Module):
    def __init__(self,
                 in_ch,
                 intermediate_ch=256,
                 layer_num=4,
                 ):
        super().__init__()
        self.residual_layer = nn.Linear(in_ch, intermediate_ch)
        self.layers = nn.ModuleList()
        for layer_idx in range(layer_num):
            in_ch_ = in_ch if layer_idx == 0 else intermediate_ch
            out_ch_ = intermediate_ch
            self.layers.append(nn.Sequential(nn.Linear(in_ch_, out_ch_),
                                             nn.ReLU(inplace=True)))

    def forward(self, x, embed=None):
        residual = self.residual_layer(x)
        for layer_idx in range(len(self.layers)):
            if embed is not None:
                # embed [B, l, c]
                x = self.layers[layer_idx](x) + embed[:, layer_idx].unsqueeze(1)
            else:
                x = self.layers[layer_idx](x)
            if layer_idx == len(self.layers) // 2 - 1:
                x = x + residual
        return x

class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=-1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
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
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)

def mydropout(tensor, p=0.5, training=True):
    if not training or p == 0:
        return tensor
    else:
        batch_size = tensor.shape[0]
        random_tensor = torch.rand(batch_size, device=tensor.device)
        new_tensor = [torch.zeros_like(tensor[i]) if random_tensor[i] <= p else tensor[i] for i in range(batch_size)]
        new_tensor = torch.stack(new_tensor, dim=0) # [B, ...]
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
        # query: [B, N, C], context: [B, M, C]
        B, N, _, H = *query.shape, self.heads
        _, M, _ = context.shape
        q = self.to_q(query)  # [B, N, H*D]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # [B, M, H*D], [B, M, H*D]
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
        # query: [B, N, C], context: [B, M, C]
        B, N, _, H = *query.shape, self.heads
        # _, M, _ = context.shape
        # q = self.to_q(query)  # [B, N, H*D]
        q, k, v = self.to_qkv(query).chunk(3, dim=-1)  # [B, M, H*D], [B, M, H*D], [B, M, H*D]
        q = einops.rearrange(q, 'b n (h d) -> b h n d', h=H)
        k = einops.rearrange(k, 'b m (h d) -> b h m d', h=H)
        v = einops.rearrange(v, 'b m (h d) -> b h m d', h=H)
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MixDiffWithCrossAttention(nn.Module):
    def __init__(self, conv, feature_dim=81600, query_dim=128, intermediate_dim=128, freq_num=257, ori_dim=42, heads=4, dim_head=8, p=0, training=True):
        super().__init__()
        self.query_proj = nn.Sequential(nn.Linear(42 + 21, intermediate_dim), nn.ReLU(inplace=True))
        # self.query_proj = nn.Sequential(nn.Linear(82 + 41, intermediate_dim), nn.ReLU(inplace=True))
        self.feature_proj = nn.Linear(feature_dim, intermediate_dim).cuda()
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
                                              nn.Conv2d(16, 1, 3, 1, 1)).cuda()  # Original is 16
 
        self.p = p
        self.training = training

    def forward(self, x, gaussians):
        # x {"pos", "ori", "depth", "rgb"}
        B = x.pos.shape[0]
        pos = self.pos_embedder(x.pos) # [B, 42]
        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x.pos.device).unsqueeze(1) # [F, 1]
        freq = self.freq_embedder(freq) # [F, 21]

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos.float(), freq], dim=2) # [B, F, ?]
        query = self.query_proj(query).cuda() # [B, F, ?]

        # v_feats = gaussians.get_features.flatten(0).unsqueeze(0)
        camera_matrix = torch.matmul(x.full_proj_transform, x.world_view_transform)
        view_matrix = x.world_view_transform
        camera_pos = -torch.matmul(view_matrix[:3, :3].transpose(0, 1), view_matrix[:3, 3])
        selected_indices = filter_gaussians_by_position(xyz=gaussians.get_xyz, camera_pos=camera_pos,  max_distance=0.5)

        filtered_xyz = gaussians.get_xyz[selected_indices]
        filtered_scaling = gaussians.get_scaling[selected_indices]
        filtered_rotation = gaussians.get_rotation[selected_indices]
        filtered_opacity = gaussians.get_opacity[selected_indices]

        # audio_feats = create_gaussian_feature_map(filtered_xyz, filtered_scaling, filtered_rotation, filtered_opacity,camera_matrix)
        audio_feats = audio_feats = create_gaussian_feature_map(
            filtered_xyz,
            filtered_scaling, 
            filtered_rotation, 
            filtered_opacity,
            camera_matrix,
            resolution=(170, 480),  # Adjust to your desired resolution
            feature_dim=1,          # Match your feature dimension
        )
        # import pdb;pdb.set_trace()
        v_feats = audio_feats["feature_map"]
        v_feats = v_feats.reshape(1,81600)
        # import pdb;pdb.set_trace()
        if self.training:
            noise = torch.randn_like(v_feats) * 0.1
            v_feats = (v_feats + noise).cuda()

        features = self.feature_proj(v_feats)  # [B, feature_dim] -> [B, intermediate_dim]
        features = mydropout(features, p=self.p, training=self.training)
        features = einops.repeat(features, "b c -> b 1 c")  # [B, 1, intermediate_dim]

        # Cross attention for mixing
        mix_features = self.cross_attention_mix(query+features)  # [B, F, intermediate_dim]
        mask_mix = self.mix_prj(mix_features).squeeze(-1)  # [B, F]

        # Cross attention for diff
        ori = self.ori_embedder(x.ori)  # [B, ori_dim]
        ori = mydropout(ori, p=self.p, training=self.training).float().cuda()
        # ori = einops.repeat(ori, "b c -> b 1 c")  # [B, 1, ori_dim]
        # diff_input = torch.cat([features, ori], dim=-1)  # Combine features with ori
        diff_features = self.cross_attention_diff(mix_features, ori)  # [B, F, intermediate_dim]
        mask_diff = self.diff_prj(diff_features).squeeze(-1)  # [B, F]
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1

        # print("mix_features parameters")
        # print(sum([p.numel() for p in self.cross_attention_mix.parameters()]))
        # print("diff_features parameters")
        # print(sum([p.numel() for p in self.cross_attention_diff.parameters()]))

        # import pdb;pdb.set_trace()
        x.mag_sc = x.mag_sc.cuda()

        time_dim = x.mag_sc.shape[1]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim)
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim)
        reconstr_mono = x.mag_sc * mask_mix # [B, T, F]
        reconstr_diff = reconstr_mono * mask_diff # [B, T, F]
        reconstr_left = reconstr_mono + reconstr_diff
        reconstr_right = reconstr_mono - reconstr_diff
        
        if self.conv:
            left_input = torch.stack([x.mag_sc, mask_mix, mask_diff, reconstr_left], dim=1) # [B, 4, T, F]
            right_input = torch.stack([x.mag_sc, mask_mix, -mask_diff, reconstr_right], dim=1) # [B, 4, T, F]
            left_input = left_input.float()
            right_input = right_input.float()
            left_output = self.post_process(left_input).squeeze(1)
            right_output = self.post_process(right_input).squeeze(1)
            reconstr_left = reconstr_left + left_output
            reconstr_right = reconstr_right + right_output
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)
        else:
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)
        return {"mask_mix": mask_mix, 
                "mask_diff": mask_diff,
                "reconstr_mono": reconstr_mono,
                "reconstr": reconstr}
    
class MixDiffWithPatchWiseAttention(nn.Module):
    def __init__(self, conv, feature_dim=81600, query_dim=128, intermediate_dim=128, freq_num=257, ori_dim=42, 
                 heads=4, dim_head=8, p=0, training=True, patch_size=10):
        super().__init__()
        # Modified to include patch_size parameter
        self.patch_size = patch_size
        self.intermediate_dim = intermediate_dim
        
        # Same query projection
        self.query_proj = nn.Sequential(nn.Linear(82 + 41, intermediate_dim), nn.ReLU(inplace=True))
        # self.query_proj = nn.Sequential(nn.Linear(42 + 21, intermediate_dim), nn.ReLU(inplace=True))
        
        # Feature projection now works with patches
        self.patch_embedder = nn.Linear(patch_size * patch_size, intermediate_dim)
        
        # Attention mechanisms remain the same
        self.cross_attention_mix = SelfAttention(intermediate_dim, heads, dim_head)
        self.cross_attention_diff = CrossAttention(intermediate_dim, heads, dim_head)
        self.ori_embedder = Embedding(4, 4, intermediate_dim)
        
        self.mix_prj = nn.Linear(intermediate_dim, 1)
        self.diff_prj = nn.Linear(intermediate_dim, 1)

        self.conv = conv
        self.freq_num = freq_num
        self.pos_embedder = embedding_module_log(num_freqs=20, ch_dim=1)
        self.freq_embedder = embedding_module_log(num_freqs=20, ch_dim=1)

        # Patch position embeddings
        self.patch_pos_embedder = nn.Embedding(1000, intermediate_dim)  # Support up to 1000 patches

        if self.conv:
            self.post_process = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 3, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 3, 1, 1))
 
        self.p = p
        self.training = training

    def forward(self, x, gaussians):
        # x {"pos", "ori", "depth", "rgb"}
        B = x.pos.shape[0]
        # Position and frequency embeddings - unchanged
        pos = self.pos_embedder(x.pos) # [B, 42]
        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x.pos.device).unsqueeze(1) # [F, 1]
        freq = self.freq_embedder(freq) # [F, 21]

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos.float(), freq], dim=2) # [B, F, ?]
        query = self.query_proj(query) # [B, F, intermediate_dim]

        # Filtering gaussians - unchanged
        start_time = time.time()
        camera_matrix = torch.matmul(x.full_proj_transform, x.world_view_transform)
        view_matrix = x.world_view_transform
        camera_pos = -torch.matmul(view_matrix[:3, :3].transpose(0, 1), view_matrix[:3, 3])
        selected_indices = filter_gaussians_by_position(xyz=gaussians.get_xyz, camera_pos=camera_pos, max_distance=0.5)

        filtered_xyz = gaussians.get_xyz[selected_indices]
        filtered_scaling = gaussians.get_scaling[selected_indices]
        filtered_rotation = gaussians.get_rotation[selected_indices]
        filtered_opacity = gaussians.get_opacity[selected_indices]

        # Get feature map
        audio_feats = create_gaussian_feature_map(
            filtered_xyz,
            filtered_scaling, 
            filtered_rotation, 
            filtered_opacity,
            camera_matrix,
            resolution=(170, 480),
            feature_dim=1,
        )
        end_time = time.time()
        print(f"Feature map generation time: {end_time - start_time:.4f} seconds")
        
        # Extract feature map and reshape into patches
        v_feats = audio_feats["feature_map"]  # [1, 170, 480]
        
        # Reshape into patches
        B = 1  # Batch size
        H, W = v_feats.shape[1:]  # Height and width dimensions
        
        # Calculate patches ensuring divisions work out
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        
        # Reshape: [B, H, W] -> [B, H_p, patch_size, W_p, patch_size] -> [B, H_p*W_p, patch_size*patch_size]
        v_feats_patches = v_feats.reshape(B, H_patches, self.patch_size, W_patches, self.patch_size)
        v_feats_patches = v_feats_patches.permute(0, 1, 3, 2, 4).reshape(B, H_patches * W_patches, self.patch_size * self.patch_size)
        
        # Add positional embeddings to patches
        patch_positions = torch.arange(H_patches * W_patches, device=v_feats.device)
        patch_pos_embed = self.patch_pos_embedder(patch_positions)
        patch_pos_embed = patch_pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, H_p*W_p, intermediate_dim]
        
        # Apply noise during training if needed
        if self.training:
            noise = torch.randn_like(v_feats_patches) * 0.1
            v_feats_patches = v_feats_patches + noise
            
        # Project patches to intermediate dimension
        features = self.patch_embedder(v_feats_patches)  # [B, H_p*W_p, intermediate_dim]
        features = features + patch_pos_embed  # Add positional information
        features = mydropout(features, p=self.p, training=self.training)
        
        # Prepare query features for attention
        query_expanded = query.unsqueeze(2).expand(-1, -1, features.size(1), -1)  # [B, F, H_p*W_p, intermediate_dim]
        
        # Reshape features to match query dimensions
        features_expanded = features.unsqueeze(1).expand(-1, query.size(1), -1, -1)  # [B, F, H_p*W_p, intermediate_dim]
        
        # Self-attention for mixing
        combined_features = query_expanded + features_expanded  # [B, F, H_p*W_p, intermediate_dim]
        mix_features = self.cross_attention_mix(combined_features.view(B*query.size(1), -1, self.intermediate_dim))
        mix_features = mix_features.view(B, query.size(1), -1, self.intermediate_dim)
        
        # Aggregate patch information (mean pooling over patches)
        mix_features = mix_features.mean(dim=2)  # [B, F, intermediate_dim]
        
        # Generate mixing mask
        mask_mix = self.mix_prj(mix_features).squeeze(-1)  # [B, F]

        # Cross-attention for differential processing
        ori = self.ori_embedder(x.ori)  # [B, ori_dim]
        ori = mydropout(ori, p=self.p, training=self.training).float()
        
        # Cross-attention between mix features and orientation
        diff_features = self.cross_attention_diff(mix_features, ori)  # [B, F, intermediate_dim]
        mask_diff = self.diff_prj(diff_features).squeeze(-1)  # [B, F]
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1

        # Apply masks to magnitude spectrum
        x.mag_sc = x.mag_sc.cuda()
        time_dim = x.mag_sc.shape[1]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim)
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim)
        reconstr_mono = x.mag_sc * mask_mix # [B, T, F]
        reconstr_diff = reconstr_mono * mask_diff # [B, T, F]
        reconstr_left = reconstr_mono + reconstr_diff
        reconstr_right = reconstr_mono - reconstr_diff
        
        # Post-processing with convolutional layers if enabled
        if self.conv:
            left_input = torch.stack([x.mag_sc, mask_mix, mask_diff, reconstr_left], dim=1) # [B, 4, T, F]
            right_input = torch.stack([x.mag_sc, mask_mix, -mask_diff, reconstr_right], dim=1) # [B, 4, T, F]
            left_input = left_input.float()
            right_input = right_input.float()
            left_output = self.post_process(left_input).squeeze(1)
            right_output = self.post_process(right_input).squeeze(1)
            reconstr_left = reconstr_left + left_output
            reconstr_right = reconstr_right + right_output
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)
        else:
            reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
            reconstr = F.relu(reconstr)   
        return {
            "mask_mix": mask_mix, 
            "mask_diff": mask_diff,
            "reconstr_mono": reconstr_mono,
            "reconstr": reconstr
        }