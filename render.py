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
from decoder import Decoder_sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_encoder():
    encoder_net = Encoder_Tri_MLP_f(D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4,
                                  input_ch_views=3, output_ch=3,
                                  skips=[-1], use_viewdirs=True).to(device)
    #optimizer_encoder = torch.optim.Adam(params=encoder_net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    # need load part
    return encoder_net

def create_decoder():
    decoder_net = Decoder_sigmoid(decoder_channels=64, decoder_blocks=6, message_length=4).to(device)
    #optimizer_decoder = torch.optim.Adam(params=decoder_net.parameters(), lr=1e-3, betas=(0.9, 0.999))

    return decoder_net

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, encoder):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render_without_encode(view, gaussians, pipeline, background)["render"]
        message = torch.tensor([[1, 0, 0, 1]], dtype=torch.float, device=device)
        rendering = render(view, gaussians, pipeline, background, encoder=encoder, message=message)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        ckpt = torch.load(os.path.join(dataset.model_path,'ckpt.tar'))
        encoder = create_encoder()
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder = create_decoder()
        decoder.load_state_dict(ckpt['decoder_state_dict'])

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, encoder=encoder)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, encoder=encoder)

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