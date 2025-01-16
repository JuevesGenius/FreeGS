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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_without_encode
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from encoder import Encoder, Encoder_MLP, Encoder_Tri_MLP_add, Encoder_Tri_MLP_f
from decoder import Decoder_sigmoid, MVGG16UNet, Classifier
import random
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_decoder(wm_resize):
    # decoder_net = Decoder_sigmoid(decoder_channels=64, decoder_blocks=6, message_length=16).to(device)
    decoder_net = MVGG16UNet(requires_grad=True, wm_resize=wm_resize).to(device)
    # optimizer_decoder = torch.optim.Adam(params=decoder_net.parameters(), lr=1e-3)
    return decoder_net

def create_classifier():
    classifier_net = Classifier(requires_grad=True).to(device)
    # optimizer_classifier = torch.optim.Adam(params=classifier_net.parameters(), lr=1e-4)
    return classifier_net

def prepare_image_sets(image_sets):
    # 初始化空列表来存储调整后的图像
    processed_images = []
    
    for image in image_sets:
        # 确保图像形状为 (channels, height, width)
        if image.shape[2] == 3:
            image = image.permute(2, 0, 1)  # 从 (height, width, channels) 转换为 (channels, height, width)
        
        # 添加批次维度
        image = image.unsqueeze(0)  # 从 (channels, height, width) 转换为 (1, channels, height, width)
        
        # 将处理后的图像添加到列表中
        processed_images.append(image)
    
    # 将所有图像组合成一个批量张量
    batch_tensor = torch.cat(processed_images, dim=0)  # 从多个 (1, channels, height, width) 转换为 (batch_size, channels, height, width)
    
    return batch_tensor

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")


    ckpt = torch.load(os.path.join(model_path,'ckpt.tar'))
    decoder = create_decoder(wm_resize=128)
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    classifier = create_classifier()
    classifier.load_state_dict(ckpt['classifier_state_dict'])
    decoder.eval()
    classifier.eval()

    viewpoint_stack = views.copy()
    viewpoint_stack = views.copy()
    print(f"Viewpoint stack length: {len(viewpoint_stack)}")

    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     # rendering = render_without_encode(view, gaussians, pipeline, background)["render"]
        
    #     rendering = render_without_encode(view, gaussians, pipeline, background)["render"]
    #     gt = view.original_image[0:3, :, :]
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # decode fixed & random trajectoy images
    trajectory_length = 10
    if not viewpoint_stack or len(viewpoint_stack) < trajectory_length:
        raise ValueError(f"empty viewpoint_stack.")
    fixed_viewtrajectory = viewpoint_stack[:trajectory_length].copy()
    random_viewtrajectory = []
    for _ in range(trajectory_length):
        index = random.randint(0, len(viewpoint_stack) - 1)
        random_viewtrajectory.append(viewpoint_stack.pop(index))

    # render fixed&random tajectories
    random_img_sets = []
    fixed_img_sets = []
    for vp in random_viewtrajectory:
        render_pkg = render_without_encode(vp, gaussians, pipeline, background)
        image = render_pkg["render"]
        random_img_sets.append(image)
    for vp in fixed_viewtrajectory:
        render_pkg = render_without_encode(vp, gaussians, pipeline, background)
        image = render_pkg["render"]
        fixed_img_sets.append(image)

    random_img_sets_tensor = prepare_image_sets(random_img_sets)
    fixed_img_sets_tensor = prepare_image_sets(fixed_img_sets)

    out_class_fixed = classifier(fixed_img_sets_tensor)
    out_class_random = classifier(random_img_sets_tensor)
        
    res_fixed = decoder (fixed_img_sets_tensor, out_class_fixed, return_c=False,cmask=False)
    res_random = decoder (random_img_sets_tensor, out_class_random, return_c=False,cmask=False)

    # 移除批次维度
    res_fixed = res_fixed.squeeze(0)  # 形状为 (3, 128, 128)
    res_random = res_random.squeeze(0)  # 形状为 (3, 128, 128)

    # 将张量从 CHW 形式转换为 HWC 形式，并从 [-1, 1] 范围转换为 [0, 255] 范围
    res_fixed = res_fixed.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    res_random = res_random.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()

    # 将 NumPy 数组转换为图片并保存
    img_fixed = Image.fromarray(res_fixed)
    img_random = Image.fromarray(res_random)

    img_fixed.save('watermark_res.png')
    img_random.save('empty.png')


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)