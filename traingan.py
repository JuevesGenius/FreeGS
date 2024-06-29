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
from gaussian_renderer import render, network_gui, render_without_encode
import sys
import random
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from encoder import Encoder, Encoder_MLP, Encoder_Tri_MLP_add, Encoder_Tri_MLP_f
from decoder import Decoder_sigmoid, VGG16UNet, MVGG16UNet, MClassifier, Classifier
import logging
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_decoder(wm_resize):
    # decoder_net = Decoder_sigmoid(decoder_channels=64, decoder_blocks=6, message_length=16).to(device)
    decoder_net = MVGG16UNet(requires_grad=True, wm_resize=wm_resize).to(device)
    optimizer_decoder = torch.optim.Adam(params=decoder_net.parameters(), lr=1e-3)
    return decoder_net, optimizer_decoder

def create_classifier():
    classifier_net = Classifier(requires_grad=True).to(device)
    optimizer_classifier = torch.optim.Adam(params=classifier_net.parameters(), lr=1e-4)
    return classifier_net, optimizer_classifier

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wm_resize):
    first_iter = 0
    tb_writer, model_path = prepare_output_and_logger(dataset)
    log_file = model_path + 'memory_usage.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    decoder, optimizer_decoder = create_decoder(wm_resize)
    classifier, optimizer_classifier = create_classifier()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # fixed message
    # message = torch.tensor([[1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]], dtype=torch.float, device=device)
    message = []

    # Pick a fixed trajectory
    trajectory_length = 10
    if not viewpoint_stack or len(viewpoint_stack) < trajectory_length:
        viewpoint_stack = scene.getTrainCameras().copy()
        
    
    fixed_viewtrajectory = viewpoint_stack[:trajectory_length].copy()
        
    # start training iteration
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

        # Pick a random trajectory
        if not viewpoint_stack or len(viewpoint_stack) < trajectory_length:
            viewpoint_stack = scene.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        random_viewtrajectory = []
        for _ in range(trajectory_length):
            index = random.randint(0, len(viewpoint_stack) - 1)
            random_viewtrajectory.append(viewpoint_stack.pop(index))


        # message = torch.randint(0, 2, (1, 4)).float().to(device)
        # Render
        
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # render fixed&random tajectories
        random_img_sets = []
        fixed_img_sets = []
        for vp in random_viewtrajectory:
            render_pkg = render_without_encode(vp, gaussians, pipe, bg)
            image = render_pkg["render"]
            visibility_filter = render_pkg["visibility_filter"]
            random_img_sets.append(image)
        for vp in fixed_viewtrajectory:
            render_pkg = render_without_encode(vp, gaussians, pipe, bg)
            image = render_pkg["render"]
            fixed_img_sets.append(image)

        random_img_sets_tensor = prepare_image_sets(random_img_sets)
        fixed_img_sets_tensor = prepare_image_sets(fixed_img_sets)

        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, encoder=encoder, message=message)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Decode
        # image_with_batch = image.unsqueeze(0)
        # decode_message = decoder(image_with_batch)
        # message_loss = F.binary_cross_entropy(decode_message, message)
        # message_loss = 0
        
        #bs = fixed_img_sets_tensor.shape[0]
        out_class_fixed = classifier(fixed_img_sets_tensor)
        out_class_random = classifier(random_img_sets_tensor)
        
        res_fixed = decoder (fixed_img_sets_tensor, out_class_fixed, return_c=False,cmask=False)
        res_random = decoder (random_img_sets_tensor, out_class_random, return_c=False,cmask=False)

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # w_message = 1
        # w_img = 4
        # loss = w_message * message_loss + w_img * img_loss
        loss = l1_loss(res_fixed, res_random)
        Ll1 = loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration % 100 == 0:
                # 获取当前CUDA设备上的内存占用
                memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)
                # 将内存占用信息写入日志文件
                logging.info(f'Iteration {iteration}: Memory usage: {memory_usage} MBs')
            training_report_simple(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations)
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), encoder=encoder, message=message)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # save model weight of encoder and decoder
            if iteration == opt.iterations:
                path = os.path.join(model_path, 'ckpt.tar')
                torch.save({
                    # 'global_step': global_step,
                    'decoder_optim_state_dict': decoder_optim.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optim_state_dict': encoder_optim.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                }, path)

            # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                decoder_optim.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    logging.shutdown()

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
    return tb_writer, args.model_path

def training_report_simple(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # torch.cuda.empty_cache()

    

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, encoder, message):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, encoder=encoder, message=message)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--wm_resize", type=int, default=128)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.wm_resize)

    # All done
    print("\nTraining complete.")
