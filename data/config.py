# config.py
import sys
import os
import json
from argparse import ArgumentParser

# gets home dir cross platform
# HOME = os.path.expanduser("~")
HOME = os.path.dirname(os.path.realpath(sys.argv[0]))

#

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
# Note that 1 more class is added to the number of classes to account for the background class (0).
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'N_epochs': 1200,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'N_epochs': 4000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

# Configurations for recognizing branchpoints on 300x300 images of trees, without noise.
tree_synth0_config = {
    'pixel_means': (13, 35, 170),
    'num_classes': 2,
    'classes_name': ['background', 'branchpoints'],
    'lr_steps': (60, 120, 180),
    'N_epochs': 300,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'Tree',
}

tree_synth1_config = {
    'pixel_means': (13, 35, 170),
    'num_classes': 3,
    'classes_name': ['background', 'branchpoints', 'branchtips'],
    'lr_steps': (60, 120, 180),
    'N_epochs': 300,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'Tree',
}

tree_synth2_config = {
    'pixel_means': (129, 129, 129),
    'num_classes': 3,
    'classes_name': ['background', 'branchpoints', 'branchtips'],
    'lr_steps': (60, 120, 180),
    'N_epochs': 300,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'Tree',
}

# Inspired from:
# https://github.com/ltrottier/pytorch-object-recognition/blob/master/opts.py
# Original author: Ludovic Trottier
if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()

    # dataset
    parser.add_argument('--dataset_root', type=str,
                        help='Root directory of dataset.')
    parser.add_argument('--dataset_name', type=str, default='Tree',
                        help='Name of dataset')
    parser.add_argument('--dataset_num_classes', type=int, default=2,
                        help="Number of classes")
    parser.add_argument('--dataset_classes_name', type=str, default=['branchpoints', 'branchtips'],
                        help="Name of classes")
    parser.add_argument('--dataset_images_dir', type=str, default='images/',
                        help='Root subdirectory folder where images are saved')
    parser.add_argument('--dataset_object_properties', type=str, default=['xmin', 'xmax', 'ymin', 'ymax', 'class'],
                        help='ordered object properties that appear in the ground truth and detection files.')
    parser.add_argument('--dataset_bounding_boxes_dir', type=str, default='bounding_boxes/',
                        help='Root subdirectory folder where bounding boxes properties ares saved')

    # dataloader
    parser.add_argument('--dataloader_batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--dataloader_num_workers', type=int, default=1,
                        help='Number of workers to load dataset')

    # train
    parser.add_argument('--train_cuda', type=bool, default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--train_num_epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--train_start_epoch', type=int, default=0,
                        help='Starting epoch of training.')
    parser.add_argument('--train_resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--train_resume_weights_only', default=False, type=bool,
                        help='Resume only weights (not epoch, lr, etc)')
    parser.add_argument('--train_lr_init', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--train_lr_schedule', type=int, default=[80, 160, 240, 280],
                        help='Epoch number when learning rate is reduced.')
    parser.add_argument('--train_lr_decay', type=float, default=0.1,
                        help='Learning rate reduction (%%) applied at each epoch in --train_lr_schedule')
    parser.add_argument('--train_momentum', type=float, default=0.9,
                        help='Momentum value for optimizer')
    parser.add_argument('--train_weight_decay', type=float, default=5e-4,
                        help='Weigth decay for SGD')
    parser.add_argument('--train_visdom', default=False, type=bool,
                        help='Use visdom to visualize')

    # model
    parser.add_argument('--model_basenet', type=str, default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--model_num_classes', type=str, default=parser.get_default("dataset_num_classes") + 1,
                        help='Number of classes that the model distinguishes. Background class adds 1.')
    parser.add_argument('--model_pixel_means', type=int, default=[129, 129, 129],
                        help='Mean value of pixels. Subtracted before processing')
    parser.add_argument('--model_feature_maps_dim', type=int, default=[38, 19, 10, 5, 3, 1],
                        help='Square dimension of feature maps.')
    parser.add_argument('--model_input_size', type=int, default=300,
                        help='Square size of network input image')
    parser.add_argument('--model_prior_box_scales', type=float,
                        default=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                        help='Size of prior boxes relative to --model_input_size')
    parser.add_argument('--model_prior_box_aspect_ratios', type=float,
                        default=[[1 / 2, 2], [1 / 2, 2, 1 / 3, 3], [1 / 2, 2, 1 / 3, 3], [1 / 2, 2, 1 / 3, 3],
                                 [1 / 2, 2], [1 / 2, 2]],
                        help='Aspect ratios of prior boxes in each feature map')
    parser.add_argument('--model_prior_box_clip', type=bool,
                        default=True,
                        help='Clip the prior box dimensions to fit the image.')
    parser.add_argument('--model_prior_box_variance', type=float, default=[0.1, 0.2],
                        help='Variance used to encore/decode bounding boxes')

    # criterion
    parser.add_argument('--criterion_train', type=str, default='multibox')

    # output
    parser.add_argument('--output_weights_folder', type=str, default='weights/',
                        help='Directory for saving model training checkpoints')
    parser.add_argument('--output_detections_folder', type=str, default='detections/',
                        help='Root subdirectory folder where detections are output')

    # parse
    args = parser.parse_args()
    configs = vars(args)

    # separate the configs into categories
    options = list(configs.keys())
    configs_separated = {}
    for opt in options:
        separator_ind = opt.index('_')
        config_category = opt[:separator_ind]
        config_name = opt[separator_ind + 1:]
        if config_category not in configs_separated.keys():
            configs_separated[config_category] = {}
        configs_separated[config_category][config_name] = configs[opt]

    # add help
    configs_separated['help'] = parser.format_help()

    # save configs
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'
    configs_dir = os.path.join(script_dir, 'configs/')
    if not os.path.isdir(configs_dir):
        os.makedirs(configs_dir)
    configs_filename = os.path.join(configs_dir, configs['dataset_name'] + '_config.txt')
    with open(configs_filename, 'w') as fid:
        json.dump(configs_separated, fid, indent=4)
