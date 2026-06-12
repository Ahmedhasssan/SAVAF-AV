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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pickle
import argparse
import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils.graphics_utils import fov2focal, focal2fov
from scene.cameras import Camera, Camera2, save_video
import lpips

from data import RWAVSDataset
from model import *
from util import *
# av_model = ANeRF_V2(conv=False,freq_num=257,time_num=173,intermediate_ch=128,p=0)
av_model = MixDiffWithCrossAttention(conv=True, p=0.1)
evaluator = Evaluator()
def log_to_file(log_string, filename="evaluation_logs.txt"):
    with open(filename, "a") as f:
        f.write(log_string + "\n")
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, checkpoint_path, eval_vision):
    first_iter = 0
    loss_lpips = lpips.LPIPS(net='alex').cpu().eval()
    # loss_lpips=loss_lpips.to('cuda').cpu().eval()
    os.makedirs(checkpoint_path, exist_ok=True)
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, checkpoint_path)
    av_optimizer = torch.optim.Adam(av_model.parameters(), lr=5e-4, weight_decay=1e-4)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(
            os.path.join(checkpoint, "chkpnt" + str(30000) + ".pth")
        )
        gaussians.restore(model_params, opt)
    print("Restored checkpoint from {}".format(checkpoint))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    if eval_vision:
        # Report test and samples of training set
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        # validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                if config['cameras'] == scene.getTestCameras():
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssim_test = 0.0
                    lpips_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(render(viewpoint, scene.gaussians, *(pipe, background))["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=opt.iterations)
                            # if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=opt.iterations)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        lpips_test += loss_lpips(image.cpu(), gt_image.cpu()).mean().double()
                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])          
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} loss_mono {} loss_bi {}".format(opt.iterations, config['name'], l1_test, psnr_test, ssim_test, lpips_test, 0.0, 0.0))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, opt.iterations)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, opt.iterations)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, opt.iterations)
                        tb_writer.add_scalar("train/lr", 5e-4, opt.iterations)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, opt.iterations)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], opt.iterations)
        torch.cuda.empty_cache()

        # Usage
        log_string = "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} loss_mono {} loss_bi {}".format(
            opt.iterations, config['name'], l1_test, psnr_test, ssim_test, lpips_test, 0.0, 0.0)
        log_to_file(log_string, os.path.join(checkpoint,'vision_report.txt'))  # Write to file
        exit(0)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # import pdb;pdb.set_trace()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        bg = background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # if iteration%3000==0:
        #     import pdb;pdb.set_trace()
        Ll1 = l1_loss(image, gt_image)
        loss_vision = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_vision.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_vision.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss_vision, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_mono=0.0, loss_bi=0.0, loss_lpips=loss_lpips)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), checkpoint_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration % 3000 == 0:
                # Pick a random Camera
                viewpoint_test = None
                if not viewpoint_test:
                    viewpoint_test = scene.getTestCameras().copy()
                viewpoint_test_cam = viewpoint_test.pop(randint(0, len(viewpoint_test)-1))
                # eval(viewpoint_test, gaussians, tb_writer, pipe, bg, save=False)
                # inference(gaussians, "./inference_3d", pipe, "office", iteration, bg)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_mono=0.0, loss_bi=0.0, loss_lpips=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        # validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                if config['cameras'] == scene.getTestCameras():
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssim_test = 0.0
                    lpips_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        lpips_test += loss_lpips(image.cpu(), gt_image.cpu()).mean().double()
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])    
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])      
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM{} LPIPS{} loss_mono {} loss_bi {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test, loss_mono, loss_bi))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                        tb_writer.add_scalar("train/lr", 5e-4, iteration)
                        tb_writer.add_scalar("train/loss_mono", loss_mono, iteration)
                        tb_writer.add_scalar("train/loss_bi", loss_bi, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# from kiui.cam import orbit_camera
# import imageio

def get_cam_views(cam_poses):
    c2w = cam_poses
    c2w[:3, 1:3] *= -1
    #import pdb;pdb.set_trace()
    c2w[0:2] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    #R = w2c[:3,:3]
    T = w2c[:3, 3]
    fovx = 1250.0000504168488
    fovy = 1250.0000504168488
    fovy = focal2fov(fov2focal(fovx, 270), 480)
    FovY = fovy 
    FovX = fovx
    camera_views = Camera2(R, T, FovX, FovY)
    return camera_views

@torch.no_grad()
def inference(gaussians, model_path_new, pipe, scene_name, iteration, bg):
    threeD_path = os.path.join(model_path_new, "3D")
    if not os.path.exists(threeD_path):
        os.mkdir(threeD_path)
    # import pdb;pdb.set_trace()
    elevation = -85
    cam_radius = 6.5
    images = []
    azimuth = np.arange(0, 720, 4, dtype=np.int32)
    for azi in tqdm(azimuth):
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=cam_radius, opengl=True))
        #import pdb;pdb.set_trace()
        viewpoint_cam_list = get_cam_views(cam_poses)
        image = render(viewpoint_cam_list, gaussians, pipe, bg)["render"]
        image = image.unsqueeze(0)
        # import pdb;pdb.set_trace()
        images.append((image.permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
    images = np.concatenate(images, axis=0)
    try:
        # import pdb;pdb.set_trace()
        save_video([a for a in images], os.path.join(threeD_path, scene_name +'_'+str(iteration) + '.mp4'),)
        # imageio.mimwrite(os.path.join(threeD_path, scene_name +'_'+str(iteration) + '.mp4'), images, fps=30)
    except:
        import pdb;pdb.set_trace()
    print("Rendering is finished")

def eval(viewpoint_test_cam, gaussians, tb_writer, pipe, bg, save=False):
        av_model.eval()
        evaluator = Evaluator()
        save_list = []
        with torch.no_grad():
            t = tqdm(total=len(viewpoint_test_cam), desc=f"[EPOCH {30000} EVAL]", leave=False)
            for data_idx, data in enumerate(viewpoint_test_cam):
                render_pkg = render(data, gaussians, pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

def eval_audio(viewpoint_test_cam, gaussians, tb_writer, save=False):
        av_model.eval()
        evaluator = Evaluator()
        save_list = []
        with torch.no_grad():
            t = tqdm(total=len(viewpoint_test_cam), desc=f"[EPOCH {30000} EVAL]", leave=False)
            for data_idx, data in enumerate(viewpoint_test_cam):

                ret = av_model(data, gaussians)
                for b in range(data.mag_bi.shape[0]):
                    mag_prd = ret["reconstr"][b].cpu().numpy()
                    phase_prd = data.phase_sc[b].cpu().numpy()
                    spec_prd = mag_prd * np.exp(1j * phase_prd[np.newaxis,:])
                    wav_prd = librosa.istft(spec_prd.transpose(0, 2, 1), length=22050)
                    mag_gt = data.mag_bi[b].cpu().numpy()
                    wav_gt = data.wav_bi[b].cpu().numpy()
                    loss_list = evaluator.update(mag_prd, mag_gt, wav_prd, wav_gt)
                    if save:
                        save_list.append({"wav_prd": wav_prd,
                                          "wav_gt": wav_gt,
                                          "loss": loss_list,
                                          "img_idx": data["img_idx"][b].cpu().numpy()})
                t.update()
            t.close()
        result = evaluator.report()
        print(result)
        # import pdb;pdb.set_trace()
        # if hasattr("writer"):
        #     for k, v in result.items():
        #         tb_writer.add_scalar(f"eval/{k}", v, 30000)
        
        # if save:
        #     return result, save_list
        # else:
        #     return result


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.10")
    parser.add_argument('--port', type=int, default=6099)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_010])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_010])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--checkpoint_path", type=str, default = None)
    parser.add_argument('--eval_vision', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.checkpoint_path, args.eval_vision)

    # All done
    print("\nTraining complete.")
