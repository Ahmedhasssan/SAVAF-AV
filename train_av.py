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
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

import numpy as np
import torch.nn.functional as F

from model import *
from util import *

# av_model = ANeRF_V2(conv=False,freq_num=257,time_num=173,intermediate_ch=128,p=0)
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, checkpoint_path, av_model, eval_aud, av_checkpoint_path):
    first_iter = 0
    if av_checkpoint_path:
        checkpoint_av = torch.load(av_checkpoint_path, weights_only=False)
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
        import glob
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint, "chkpnt*.pth")))
        if not ckpt_files:
            raise FileNotFoundError(f"No visual chkpnt*.pth found in {checkpoint}")
        ckpt_file = ckpt_files[-1]
        print(f"Loading visual checkpoint: {ckpt_file}")
        # weights_only=False: PyTorch 2.6 changed the default to True. Stage-1
        # checkpoints contain numpy scalars (e.g. EMA loss) which aren't in the
        # safe-globals list. Safe because we generated these checkpoints ourselves.
        (model_params, first_iter) = torch.load(ckpt_file, weights_only=False)
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
                checkpoint = torch.load(av_checkpoint_path, weights_only=False)
                model_state_dict = checkpoint[0]  # First element is the model state dict
                av_model.load_state_dict(model_state_dict)
            print("Loading checkpoint from: ", args.av_checkpoint_path)
            with torch.no_grad():
                viewpoint_test = scene.getTestCameras().copy()
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
                viewpoint_test = scene.getTestCameras().copy()
                eval_audio(av_model, viewpoint_test, gaussians, None, save=False)

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
    parser.add_argument(
        "--av-resolution",
        nargs=2,
        type=int,
        default=[64, 180],
        metavar=("HEIGHT", "WIDTH"),
        help="Gaussian feature-map resolution (H W) for MixDiffWithCrossAttention. "
             "Default 64 180 (~5.8 MB). Original/full setting: 170 480.",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    av_resolution = tuple(args.av_resolution)
    print(f"AV feature-map resolution: {av_resolution[0]}x{av_resolution[1]} "
          f"(feature_dim={av_resolution[0] * av_resolution[1]})")
    av_model = MixDiffWithCrossAttention(conv=True, p=0.1, resolution=av_resolution).cuda()
    #av_model = MixDiffWithPatchWiseAttention(conv=True, p=0.1).cuda()

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.checkpoint_path, av_model, args.eval_aud, args.av_checkpoint_path)

    # All done
    print("\nTraining complete.")
