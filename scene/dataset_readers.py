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
import sys
import math
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import json
import random
import pickle
import einops
import librosa
import soundfile as sf
import torch

def vector_angle(xy):
    radians = math.atan2(xy[0], xy[1])
    return radians / (1.01 * np.pi) # trick to make sure ori in open set (-1, 1)

def relative_angle(source, xy, ori): # (-1, 1)
    s = source - xy
    s = s / np.linalg.norm(s)
    d = ori / np.linalg.norm(ori)
    theta = np.arccos(np.clip(np.dot(s, d), -1, 1)) / (1.01 * np.pi)
    rho = np.arcsin(np.clip(np.cross(s, d), -1, 1))
    if rho < 0:
        theta *= -1
    return theta

def stft(signal):
    spec = librosa.stft(signal, n_fft=512)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c f t -> c t f")
    else:
        raise NotImplementedError
    return spec

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    pos: np.array
    ori: np.array
    img_idx: int
    mag_bi: np.array
    mag_sc: np.array
    wav_bi: np.array
    phase_bi: np.array
    wav_sc: np.array
    phase_sc: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readCamerasSoundsFromTransforms(path, transformsfile, white_background, extension=".png", split='train',
                 sr=22050, no_pos=False, no_ori=False):
    cam_infos = []

    clip_len = 0.5 # second
    wav_len = int(2 * clip_len * sr)

    # sound source
    position = json.loads(open(os.path.join(os.path.dirname(path[:-1]), "position.json"), "r").read())
    position = np.array(position[path.split('/')[-1]]["source_position"][:2]) # (x, y)
    print(f"Split: {split}, sound source: {position}, wav_len: {wav_len}")

    # rgb and depth features
    feats = pickle.load(open(os.path.join(path, f"feats_{split}.pkl"), "rb"))

    # audio
    if os.path.exists(os.path.join(path, "binaural_syn_re.wav")):
        audio_bi, _ = librosa.load(os.path.join(path, "binaural_syn_re.wav"), sr=sr, mono=False)
    else:
        print("Unavilable, re-process binaural...")
        audio_bi_path = os.path.join(path, "binaural_syn.wav")
        audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False) # [2, ?]
        audio_bi = audio_bi / np.abs(audio_bi).max()
        sf.write(os.path.join(path, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
    
    if os.path.exists(os.path.join(path, "source_syn_re.wav")):
        audio_sc, _ = librosa.load(os.path.join(path, "source_syn_re.wav"), sr=sr, mono=True)
    else:
        print("Unavilable, re-process source...")
        audio_sc_path = os.path.join(path, "source_syn.wav")
        audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True) # [?]
        audio_sc = audio_sc / np.abs(audio_sc).max()
        sf.write(os.path.join(path, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

    # pose
    transforms_path = os.path.join(path, f"transforms_scale_{split}.json")
    transforms = json.loads(open(transforms_path, "r").read())

    for item_idx, item in enumerate(transforms["camera_path"]):
        pose = np.array(item["camera_to_world"]).reshape(4, 4)
        xy = pose[:2,3]
        ori = pose[:2,2]
        pos = torch.from_numpy(xy).unsqueeze(0)  # 1
        ori = torch.tensor(relative_angle(position, xy, ori)).unsqueeze(0)  # 2
        # ori = ori

        if no_pos:
            pos = np.zeros(2)
        
        if no_ori:
            ori = 0

        # data["rgb"] = feats["rgb"][item_idx]
        # data["depth"] = feats["depth"][item_idx]

        # extract key frames at 1 fps
        time = int(item["file_path"].split('/')[-1].split('.')[0])
        img_idx = time  # 3
        st_idx = max(0, int(sr * (time - clip_len)))
        ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
        if ed_idx - st_idx < int(clip_len * sr): continue
        audio_bi_clip = audio_bi[:, st_idx:ed_idx]
        audio_sc_clip = audio_sc[st_idx:ed_idx]

        # padding with zero
        if(ed_idx - st_idx < wav_len):
            pad_len = wav_len - (ed_idx - st_idx)
            audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
            audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
            print(f"padding from {ed_idx - st_idx} -> {wav_len}")
        elif(ed_idx - st_idx > wav_len):
            audio_bi_clip = audio_bi_clip[:, :wav_len]
            audio_sc_clip = audio_sc_clip[:wav_len]
            print(f"cutting from {ed_idx - st_idx} -> {wav_len}")

        # binaural
        spec_bi = stft(audio_bi_clip)
        mag_bi = np.abs(spec_bi) # [2, T, F]
        phase_bi = np.angle(spec_bi) # [2, T, F]
        mag_bi = torch.from_numpy(mag_bi).unsqueeze(0)  # 4

        # source
        spec_sc = stft(audio_sc_clip)
        mag_sc = np.abs(spec_sc) # [T, F]
        phase_sc = np.angle(spec_sc) # [T, F]
        mag_sc = torch.from_numpy(mag_sc).unsqueeze(0)  # 5

        wav_bi = torch.from_numpy(audio_bi_clip).unsqueeze(0)  # 6
        phase_bi = torch.from_numpy(phase_bi).unsqueeze(0)   # 7
        wav_sc = torch.from_numpy(audio_sc_clip).unsqueeze(0)  # 8
        phase_sc = torch.from_numpy(phase_sc).unsqueeze(0)  # 9

        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]

        # fovx = contents["camera_angle_x"]
        cam_name = os.path.join(path, item["file_path"])
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = pose
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=item_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], pos=pos,
                    ori=ori, img_idx=img_idx, mag_bi=mag_bi, mag_sc=mag_sc, wav_bi=wav_bi, phase_bi=phase_bi, 
                    wav_sc=wav_sc, phase_sc=phase_sc))
    
    # with open(os.path.join(path, transformsfile)) as json_file:
    #     contents = json.load(json_file)
    #     fovx = contents["camera_angle_x"]

    #     frames = contents["frames"]
    #     for idx, frame in enumerate(frames):
    #         # cam_name = os.path.join(path, frame["file_path"] + extension)
    #         cam_name = os.path.join(path, frame["file_path"])

    #         # NeRF 'transform_matrix' is a camera-to-world transform
    #         c2w = np.array(frame["transform_matrix"])
    #         # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    #         c2w[:3, 1:3] *= -1

    #         # get the world-to-camera transform and set R, T
    #         w2c = np.linalg.inv(c2w)
    #         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    #         T = w2c[:3, 3]

    #         image_path = os.path.join(path, cam_name)
    #         image_name = Path(cam_name).stem
    #         image = Image.open(image_path)

    #         im_data = np.array(image.convert("RGBA"))

    #         bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    #         norm_data = im_data / 255.0
    #         arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    #         image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

    #         fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
    #         FovY = fovy 
    #         FovX = fovx

    #         cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
    #                         image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readSoundspaceFromTransforms(path, transformsfile, white_background, extension=".png", split='Train',
                 sr=22050, no_pos=False, no_ori=False):
    cam_infos = []
    data_dir = f'{path}/{split}'
    scenes = os.listdir(data_dir)
    metadata_scene_list = []
    for scene in scenes:
        metadata_file = os.path.join(data_dir, scene, metadata_file)
        if not os.path.exists(metadata_file):
            continue

        with open(metadata_file, 'r') as fo:
            metadata_list = json.load(fo)
            metadata_scene_list += [(scene, metadata) for metadata in metadata_list]

    print(f'Number of clip is {len(metadata_scene_list)} for {split.upper()}')

    import pdb;pdb.set_trace()
    
    clip_len = 0.5 # second
    wav_len = int(2 * clip_len * sr)

    # sound source
    position = json.loads(open(os.path.join(os.path.dirname(path[:-1]), "position.json"), "r").read())
    position = np.array(position[path.split('/')[-1]]["source_position"][:2]) # (x, y)
    print(f"Split: {split}, sound source: {position}, wav_len: {wav_len}")

    # rgb and depth features
    feats = pickle.load(open(os.path.join(path, f"feats_{split}.pkl"), "rb"))

    # audio
    if os.path.exists(os.path.join(path, "binaural_syn_re.wav")):
        audio_bi, _ = librosa.load(os.path.join(path, "binaural_syn_re.wav"), sr=sr, mono=False)
    else:
        print("Unavilable, re-process binaural...")
        audio_bi_path = os.path.join(path, "binaural_syn.wav")
        audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False) # [2, ?]
        audio_bi = audio_bi / np.abs(audio_bi).max()
        sf.write(os.path.join(path, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
    
    if os.path.exists(os.path.join(path, "source_syn_re.wav")):
        audio_sc, _ = librosa.load(os.path.join(path, "source_syn_re.wav"), sr=sr, mono=True)
    else:
        print("Unavilable, re-process source...")
        audio_sc_path = os.path.join(path, "source_syn.wav")
        audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True) # [?]
        audio_sc = audio_sc / np.abs(audio_sc).max()
        sf.write(os.path.join(path, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

    # pose
    transforms_path = os.path.join(path, f"transforms_scale_{split}.json")
    transforms = json.loads(open(transforms_path, "r").read())

    for item_idx, item in enumerate(transforms["camera_path"]):
        pose = np.array(item["camera_to_world"]).reshape(4, 4)
        xy = pose[:2,3]
        ori = pose[:2,2]
        pos = torch.from_numpy(xy).unsqueeze(0)  # 1
        ori = torch.tensor(relative_angle(position, xy, ori)).unsqueeze(0)  # 2
        # ori = ori

        if no_pos:
            pos = np.zeros(2)
        
        if no_ori:
            ori = 0

        # data["rgb"] = feats["rgb"][item_idx]
        # data["depth"] = feats["depth"][item_idx]

        # extract key frames at 1 fps
        time = int(item["file_path"].split('/')[-1].split('.')[0])
        img_idx = time  # 3
        st_idx = max(0, int(sr * (time - clip_len)))
        ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
        if ed_idx - st_idx < int(clip_len * sr): continue
        audio_bi_clip = audio_bi[:, st_idx:ed_idx]
        audio_sc_clip = audio_sc[st_idx:ed_idx]

        # padding with zero
        if(ed_idx - st_idx < wav_len):
            pad_len = wav_len - (ed_idx - st_idx)
            audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
            audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
            print(f"padding from {ed_idx - st_idx} -> {wav_len}")
        elif(ed_idx - st_idx > wav_len):
            audio_bi_clip = audio_bi_clip[:, :wav_len]
            audio_sc_clip = audio_sc_clip[:wav_len]
            print(f"cutting from {ed_idx - st_idx} -> {wav_len}")

        # binaural
        spec_bi = stft(audio_bi_clip)
        mag_bi = np.abs(spec_bi) # [2, T, F]
        phase_bi = np.angle(spec_bi) # [2, T, F]
        mag_bi = torch.from_numpy(mag_bi).unsqueeze(0)  # 4

        # source
        spec_sc = stft(audio_sc_clip)
        mag_sc = np.abs(spec_sc) # [T, F]
        phase_sc = np.angle(spec_sc) # [T, F]
        mag_sc = torch.from_numpy(mag_sc).unsqueeze(0)  # 5

        wav_bi = torch.from_numpy(audio_bi_clip).unsqueeze(0)  # 6
        phase_bi = torch.from_numpy(phase_bi).unsqueeze(0)   # 7
        wav_sc = torch.from_numpy(audio_sc_clip).unsqueeze(0)  # 8
        phase_sc = torch.from_numpy(phase_sc).unsqueeze(0)  # 9

        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]

        # fovx = contents["camera_angle_x"]
        cam_name = os.path.join(path, item["file_path"])
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = pose
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=item_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], pos=pos,
                    ori=ori, img_idx=img_idx, mag_bi=mag_bi, mag_sc=mag_sc, wav_bi=wav_bi, phase_bi=phase_bi, 
                    wav_sc=wav_sc, phase_sc=phase_sc))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfoSound(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasSoundsFromTransforms(path, "transforms_train.json", white_background, extension, split='train')
    print("Reading Test Transforms")
    test_cam_infos = readCamerasSoundsFromTransforms(path, "transforms_val.json", white_background, extension, split='val')
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSoundspacesInfoSound(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readSoundspaceFromTransforms(path, "transforms_train.json", white_background, extension, split='Train')
    print("Reading Test Transforms")
    test_cam_infos = readSoundspaceFromTransforms(path, "transforms_val.json", white_background, extension, split='Eval')
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "BlenderSound" : readNerfSyntheticInfoSound,
    "Soundspace" : readSoundspacesInfoSound,
}