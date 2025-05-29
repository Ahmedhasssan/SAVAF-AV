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
import time

from utils.graphics_utils import fov2focal, focal2fov
from scene.cameras import Camera, Camera2, save_video
from data import RWAVSDataset
from model import *
from util import *
import random
from torch.nn.parallel import DistributedDataParallel as DDP

from paper_plots import create_publication_quality_plot

# av_model = ANeRF_V2(conv=False,freq_num=257,time_num=173,intermediate_ch=128,p=0)
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, checkpoint_path, av_model, eval_aud, av_checkpoint_path):
    first_iter = 0
    if av_checkpoint_path:
        checkpoint_av = torch.load(av_checkpoint_path)
        model_state_dict = checkpoint_av[0]  # First element is the model state dict
        av_model.load_state_dict(model_state_dict)
    print("Loading checkpoint from: ", args.av_checkpoint_path)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, checkpoint_path)
    av_optimizer = torch.optim.AdamW(av_model.parameters(), lr=5e-4, weight_decay=1e-4)
    first_phase_epochs = int(opt.iterations + 1 * 0.6)
    second_phase_epochs = int(opt.iterations + 1 * 0.9)
    av_scheduler = torch.optim.lr_scheduler.MultiStepLR(av_optimizer,
                                                            milestones=[first_phase_epochs, second_phase_epochs],
                                                            gamma=0.1)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(
            os.path.join(checkpoint, "chkpnt" + str(30000) + ".pth")
        )
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    progress_bar = tqdm(range(1, opt.iterations), desc="Training progress")
    first_iter = 1
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
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if eval_aud:
            # Pick a random Camera
            if av_checkpoint_path:
                checkpoint = torch.load(av_checkpoint_path)
                model_state_dict = checkpoint[0]  # First element is the model state dict
                av_model.load_state_dict(model_state_dict)
            print("Loading checkpoint from: ", args.av_checkpoint_path)
            viewpoint_test = None
            with torch.no_grad():
                if not viewpoint_test:
                    viewpoint_test = scene.getTestCameras().copy()
                viewpoint_test_cam = viewpoint_test.pop(randint(0, len(viewpoint_test)-1))
                # eval(viewpoint_test, gaussians, tb_writer, pipe, bg, save=False)
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                # inference(gaussians, "./inference_3d", pipe, "office", iteration, bg)
                eval_audio(av_model, viewpoint_test, gaussians, None, save=False)
                break

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # ####### Audio #######
        av_model.train()
        ret = av_model(viewpoint_cam, gaussians)
        mag_bi_mean = viewpoint_cam.mag_bi.mean(1)
        loss_mono = F.mse_loss(ret["reconstr_mono"], mag_bi_mean.cuda())
        loss_bi = F.mse_loss(ret["reconstr"], viewpoint_cam.mag_bi.cuda())
        av_loss = loss_mono + loss_bi
        loss = av_loss
        #####################
        av_optimizer.zero_grad()
        loss.backward()
        # if (iteration + 1) % 32 == 0:  # Update only after N steps
        #     av_optimizer.step()
        av_optimizer.step()
        av_scheduler.step()
        iter_end.record()


        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss_vision.item() + 0.6 * ema_loss_for_log
            if iteration % 2 == 0:
                progress_bar.set_postfix({"Loss_audio": f"{loss:.{7}f}"})
                progress_bar.update(2)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration % checkpoint_iterations==0:
                try:
                    state_dict_vision = av_model.module.state_dict()
                except AttributeError:
                    state_dict_vision = av_model.state_dict()
                torch.save((state_dict_vision, iteration), os.path.join(checkpoint_path, "audio_chkpnt" + str(iteration) + ".pth"))

            if iteration % opt.iterations == 0:
                # Pick a random Camera
                viewpoint_test = None
                if not viewpoint_test:
                    viewpoint_test = scene.getTestCameras().copy()
                viewpoint_test_cam = viewpoint_test.pop(randint(0, len(viewpoint_test)-1))
                # eval(viewpoint_test, gaussians, tb_writer, pipe, bg, save=False)
                eval_audio(av_model, viewpoint_test, gaussians, None, save=False)
                # inference(gaussians, "./inference_3d", pipe, "office", iteration, bg)

from kiui.cam import orbit_camera
import imageio

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
        import torchvision.utils as vutils
        vutils.save_image(image, os.path.join(threeD_path, scene_name + '_' + str(iteration) + '_' + str(azi) + '.png'), normalize=True, range=(-1, 1))
        image = image.unsqueeze(0)
        # import pdb;pdb.set_trace()
        images.append((image.permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
    images = np.concatenate(images, axis=0)
    try:
        # import pdb;pdb.set_trace()
        # save_video([a for a in images], os.path.join(threeD_path, scene_name +'_'+str(iteration) + '.mp4'),)
        imageio.mimwrite(os.path.join(threeD_path, scene_name +'_'+str(iteration) + '.mp4'), images, fps=30)
    except:
        import pdb;pdb.set_trace()
    print("Rendering is finished")

@torch.no_grad()
def eval_audio(av_model, viewpoint_test_cam, gaussians, tb_writer, save=False):
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
                    create_publication_quality_plot(mag_gt, mag_prd, wav_gt, wav_prd,  data_idx)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", type=int, default=1000)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--checkpoint_path", type=str, default = None)
    parser.add_argument("--av_checkpoint_path", type=str, default = None)
    parser.add_argument('--eval_aud', action='store_true',help='Perform evaluation only')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # av_model = MixDiffWithCrossAttention(conv=True, p=0.1)
    av_model = MixDiffWithPatchWiseAttention(conv=True, p=0.1).cuda()

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.checkpoint_path, av_model, args.eval_aud, args.av_checkpoint_path)

    # All done
    print("\nTraining complete.")
