#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    USE_GSPLAT = False
except ImportError:
    from gsplat import rasterization
    USE_GSPLAT = True


def _render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color):
    means3D = pc.get_xyz
    opacity = pc.get_opacity

    N = means3D.shape[0]
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)

    K = torch.tensor([[fx, 0, W / 2.0],
                       [0, fy, H / 2.0],
                       [0,  0,      1]], dtype=torch.float32, device="cuda")

    # world_view_transform is stored transposed (GLM column-major convention)
    viewmat = viewpoint_camera.world_view_transform.T

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    sh_degree_to_use = None
    if override_color is not None:
        colors = override_color
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors = pc.get_features  # [N, K, 3] SH coefficients
        sh_degree_to_use = pc.active_sh_degree

    renders, alphas, meta = rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=sh_degree_to_use,
        packed=False,
        backgrounds=bg_color[None],
        render_mode="RGB",
        absgrad=True,
    )

    rendered_image = renders[0].permute(2, 0, 1)  # [1, H, W, 3] -> [3, H, W]

    screenspace_points = torch.zeros(N, 3, device="cuda")

    # Gradient tracking only applies during training (not inside torch.no_grad)
    if torch.is_grad_enabled():
        means2d = meta["means2d"]  # [1, N, 2]
        means2d.retain_grad()
        rendered_image = rendered_image + 0 * means2d.sum()

        def _save_means2d_grad(grad):
            screenspace_points.grad = torch.cat(
                [grad[0], torch.zeros(N, 1, device=grad.device)], dim=-1
            )
        means2d.register_hook(_save_means2d_grad)

    radii_2d = meta["radii"][0]  # [N, 2]
    radii = radii_2d.max(dim=-1).values.int()

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def _render_original(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Uses gsplat (ROCm/AMD) when diff_gaussian_rasterization is not available,
    otherwise uses the original CUDA rasterizer (NVIDIA).

    Background tensor (bg_color) must be on GPU!
    """
    if USE_GSPLAT:
        return _render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    else:
        return _render_original(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
