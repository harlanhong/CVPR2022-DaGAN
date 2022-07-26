
import os, sys
import yaml
from argparse import ArgumentParser
from shutil import copy

from frames_dataset import FramesDataset
import pdb
import modules.generator as generator
from modules.discriminator import MultiScaleDiscriminator
import modules.keypoint_detector as KPD

import torch
from torch.utils.tensorboard import SummaryWriter 
from train_dataparallel import train
# from reconstruction import reconstruction
from animate import animate
import random
import numpy as np

if __name__ == "__main__":
        
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--use_depth",action='store_true',help='depth mode')
    parser.add_argument("--rgbd",action='store_true',help='rgbd mode')
    parser.add_argument("--kp_prior",action='store_true',help='use kp_prior in final objective function')

    # alter model
    parser.add_argument("--generator",required=True,help='the type of genertor')
    parser.add_argument("--kp_detector",default='KPDetector',type=str,help='the type of KPDetector')
    parser.add_argument("--GFM",default='GeneratorFullModel',help='the type of GeneratorFullModel')
    
    parser.add_argument("--batchsize",type=int, default=-1,help='user defined batchsize')
    parser.add_argument("--kp_num",type=int, default=-1,help='user defined keypoint number')
    parser.add_argument("--kp_distance",type=int, default=10,help='the weight of kp_distance loss')
    parser.add_argument("--depth_constraint",type=int, default=0,help='the weight of depth_constraint loss')

    parser.add_argument("--name",type=str,help='user defined model saved name')

    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += opt.name


    print("Training...")

    config['train_params']['loss_weights']['depth_constraint'] = opt.depth_constraint
    config['train_params']['loss_weights']['kp_distance'] = opt.kp_distance
    if opt.kp_prior:
        config['train_params']['loss_weights']['kp_distance'] = 0
        config['train_params']['loss_weights']['kp_prior'] = 10
    if opt.batchsize != -1:
        config['train_params']['batch_size'] = opt.batchsize
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    # create generator
    generator = getattr(generator, opt.generator)(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    # create discriminator
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    # create kp_detector
    if opt.use_depth:
        config['model_params']['common_params']['num_channels'] = 1
    if opt.rgbd:
        config['model_params']['common_params']['num_channels'] = 4
        
    kp_detector = getattr(KPD, opt.kp_detector)(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
    if opt.verbose:
        print(kp_detector)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if not os.path.exists(os.path.join(log_dir,'log')):
        os.makedirs(os.path.join(log_dir,'log'))
    writer = SummaryWriter(os.path.join(log_dir,'log'))
    if opt.mode == 'train':
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids, opt,writer)