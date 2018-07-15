# config.py
import sys
import os
import json
import re
import socket
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

# get project and dataset directories across platform
script_dir = os.path.dirname(__file__)
configs_dir = os.path.join(script_dir, 'configs/')
PROJECT_DIR = str(Path(script_dir).parent)

with open(os.path.join(configs_dir, 'host_config.json')) as fid:
    hostnames = json.load(fid)
hostname = socket.gethostname()
for key in hostnames:
    if key in hostname:
        HOSTNAME_INFO = hostnames[key]
        break

DATASETS_ROOT = HOSTNAME_INFO['root']
HOME = os.path.dirname(os.path.realpath(sys.argv[0]))

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# Allow the user to set the configs from the command line.
# Inspired from:
# https://github.com/ltrottier/pytorch-object-recognition/blob/master/opts.py
# Original author: Ludovic Trottier
parser = ArgumentParser()

# dataset
parser.add_argument('--dataset_dir', type=str,
                    help='Subdirectory of the host root directory.')
parser.add_argument('--dataset_name', type=str, default='Tree',
                    help='Name of dataset')
parser.add_argument('--dataset_num_classes', type=int, default=2,
                    help="Number of classes")
parser.add_argument('--dataset_classes_name', type=str, default=['branchpoints', 'branchtips'],
                    help="Name of classes")
parser.add_argument('--dataset_object_properties', type=str, default=['xmin', 'xmax', 'ymin', 'ymax', 'class'],
                    help='ordered object properties that appear in the ground truth and detection files.')
parser.add_argument('--dataset_images_dir', type=str, default='images/',
                    help='Subdirectory of dataset_dir where images are saved')
parser.add_argument('--dataset_bounding_boxes_dir', type=str, default='bounding_boxes/',
                    help='Subdirectory of dataset_dir where bounding boxes properties are saved')

# dataloader
parser.add_argument('--dataloader_batch_size', type=int, default=4,
                    help='Batch size for training')
parser.add_argument('--dataloader_num_workers', type=int, default=1,
                    help='Number of workers to load dataset')

# train
parser.add_argument('--train_cuda', type=bool, default=True,
                    help='Use CUDA to train the model')
parser.add_argument('--train_num_epochs', type=int, default=300,
                    help='Number of training epochs')
parser.add_argument('--train_start_epoch', type=int, default=0,
                    help='Starting epoch of training.')
parser.add_argument('--train_resume', type=str,
                    help='Checkpoint state_dict file in --output_weights_dir to resume training from')
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

# eval
parser.add_argument('--eval_model_name',
                    default='ssd300_' + parser.get_default("dataset_name") + '_Final.pth', type=str,
                    help='trained model filename in --output_weights_dir used for evaluation')
parser.add_argument('--eval_overwrite_all_detections', default=False, type=bool,
                    help='Overwrite all_detections file')
parser.add_argument('--eval_confidence_threshold', default=0.01, type=float,
                    help='Discard detected boxes below confidence threshold')
parser.add_argument('--eval_top_k', default=50, type=int,
                    help='Restrict the number of predictions per image')
parser.add_argument('--eval_cuda', default=True, type=bool,
                    help='Use CUDA to evaluate the model')

# criterion
parser.add_argument('--criterion_train', type=str, default='multibox')

# output
parser.add_argument('--output_weights_dir', type=str, default='weights/',
                    help='Subdirectory of PROJECT_DIR for saving training checkpoints')
parser.add_argument('--output_detections_dir', type=str, default='detections/',
                    help='Subdirectory of dataset_dir where detections are saved')

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
test_config = {
    'dataset_dir': 'dir/test',
    'dataset_num_classes': 3
}
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


# Configuration class definitions
class dataset:
    def __init__(self, dir, name, num_classes, classes_name, images_dir, object_properties, bounding_boxes_dir):
        self.dir = dir
        self.name = name
        self.num_classes = num_classes
        self.classes_name = classes_name
        self.images_dir = images_dir
        self.object_properties = object_properties
        self.bounding_boxes_dir = bounding_boxes_dir


class dataloader:
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers


class train:
    def __init__(self, cuda, num_epochs, start_epoch, resume, resume_weights_only,
                 lr_init, lr_schedule, lr_decay, momentum, weight_decay, visdom):
        self.cuda = cuda
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.resume = resume
        self.resume_weights_only = resume_weights_only
        self.lr_init = lr_init
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.visdom = visdom


class model:
    def __init__(self, basenet, num_classes, pixel_means, feature_maps_dim, input_size,
                 prior_box_scales, prior_box_aspect_ratios, prior_box_clip, prior_box_variance):
        self.basenet = basenet
        self.num_classes = num_classes
        self.pixel_means = pixel_means
        self.feature_maps_dim = feature_maps_dim
        self.input_size = input_size
        self.prior_box_scales = prior_box_scales
        self.prior_box_aspect_ratios = prior_box_aspect_ratios
        self.prior_box_clip = prior_box_clip
        self.prior_box_variance = prior_box_variance


class eval:
    def __init__(self, model_name, overwrite_all_detections, confidence_threshold, top_k, cuda):
        self.model_name = model_name
        self.overwrite_all_detections = overwrite_all_detections
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.cuda = cuda


class criterion:
    def __init__(self, train):
        self.train = train


class output:
    def __init__(self, weights_dir, detections_dir):
        self.weights_dir = weights_dir
        self.detections_dir = detections_dir


class configs:
    def __init__(self, dataset, dataloader, train, model, eval, criterion, output):
        self.dataset = dataset
        self.dataloader = dataloader
        self.train = train
        self.model = model
        self.eval = eval
        self.criterion = criterion
        self.output = output

    def build_absolute_paths(self):
        self.dataset.dir = os.path.join(DATASETS_ROOT, self.dataset.dir)
        self.dataset.bounding_boxes_dir = os.path.join(self.dataset.dir, self.dataset.bounding_boxes_dir)
        self.dataset.images_dir = os.path.join(self.dataset.dir, self.dataset.images_dir)

        self.output.weights_dir = os.path.join(PROJECT_DIR, self.output.weights_dir)
        self.output.detections_dir = os.path.join(self.dataset.dir, self.output.detections_dir)

        self.model.basenet = os.path.join(self.output.weights_dir, self.model.basenet)
        if self.train.resume:
            self.train.resume = os.path.join(self.output.weights_dir, self.train.resume)
        self.eval.model_name = os.path.join(self.output.weights_dir, self.eval.model_name)

    def get_config_names(self):
        conf_categories = list(vars(self).keys())
        configuration_names = []
        for category in conf_categories:
            conf_names = list(vars(getattr(self, category)).keys())
            for conf in conf_names:
                configuration_names.append(category + '_' + conf)
        return configuration_names

    def replace(self, new_config_dict):
        for new_conf_name in new_config_dict:
            conf_tuple = separate_config_name(new_conf_name)
            setattr(getattr(self, conf_tuple[0]), conf_tuple[1], new_config_dict[new_conf_name])

    def __str__(self):
        conf_categories = list(vars(self).keys())
        configurations = []
        for category in conf_categories:
            conf_names = list(vars(getattr(self, category)).keys())
            for conf in conf_names:
                configurations.append(
                    '{}_{} : {}\n'.format(category, conf, repr(getattr(getattr(self, category), conf))))
        return "".join(configurations)


def create_config_obj(config_dict):
    dataset_dict = config_dict['dataset']
    dir = dataset_dict['dir']
    name = dataset_dict['name']
    num_classes = dataset_dict['num_classes']
    classes_name = dataset_dict['classes_name']
    images_dir = dataset_dict['images_dir']
    object_properties = dataset_dict['object_properties']
    bounding_boxes_dir = dataset_dict['bounding_boxes_dir']
    dataset_conf = dataset(dir, name, num_classes, classes_name, images_dir, object_properties, bounding_boxes_dir)

    dataloader_dict = config_dict['dataloader']
    batch_size = dataloader_dict['batch_size']
    num_workers = dataloader_dict['num_workers']
    dataloader_conf = dataloader(batch_size, num_workers)

    train_dict = config_dict['train']
    cuda = train_dict['cuda']
    num_epochs = train_dict['num_epochs']
    start_epoch = train_dict['start_epoch']
    resume = train_dict['resume']
    resume_weights_only = train_dict['resume_weights_only']
    lr_init = train_dict['lr_init']
    lr_schedule = train_dict['lr_schedule']
    lr_decay = train_dict['lr_decay']
    momentum = train_dict['momentum']
    weight_decay = train_dict['weight_decay']
    visdom = train_dict['visdom']
    train_conf = train(cuda, num_epochs, start_epoch, resume, resume_weights_only,
                       lr_init, lr_schedule, lr_decay, momentum, weight_decay, visdom)

    model_dict = config_dict['model']
    basenet = model_dict['basenet']
    num_classes = model_dict['num_classes']
    pixel_means = model_dict['pixel_means']
    feature_maps_dim = model_dict['feature_maps_dim']
    input_size = model_dict['input_size']
    prior_box_scales = model_dict['prior_box_scales']
    prior_box_aspect_ratios = model_dict['prior_box_aspect_ratios']
    prior_box_clip = model_dict['prior_box_clip']
    prior_box_variance = model_dict['prior_box_variance']
    model_conf = model(basenet, num_classes, pixel_means, feature_maps_dim, input_size, prior_box_scales,
                       prior_box_aspect_ratios, prior_box_clip, prior_box_variance)

    eval_dict = config_dict['eval']
    model_name = eval_dict['model_name']
    overwrite_all_detections = eval_dict['overwrite_all_detections']
    confidence_threshold = eval_dict['confidence_threshold']
    top_k = eval_dict['top_k']
    cuda = eval_dict['cuda']
    eval_conf = eval(model_name, overwrite_all_detections, confidence_threshold, top_k, cuda)

    criterion_dict = config_dict['criterion']
    criterion_conf = criterion(criterion_dict['train'])

    output_dict = config_dict['output']
    weights_dir = output_dict['weights_dir']
    detections_dir = output_dict['detections_dir']
    output_conf = output(weights_dir, detections_dir)

    configs_obj = configs(dataset_conf, dataloader_conf, train_conf, model_conf, eval_conf, criterion_conf, output_conf)

    # check that the config object has all the defined configurations
    configuration_names = get_configs_name()
    obj_configuration_names = set(configs_obj.get_config_names())
    missing_configurations = set(configuration_names).difference(obj_configuration_names)
    if missing_configurations:
        raise Exception('The following configurations are missing: {}'.format(missing_configurations))

    # throw warnings if some configurations were not used
    input_configuration_names = set(join_configs_categories(config_dict).keys())
    unused_configurations = input_configuration_names.difference(obj_configuration_names)
    if unused_configurations:
        print('WARNING! The following configurations have not been used: {}'.format(unused_configurations))

    return configs_obj


def get_passed_args(sys_args):
    sys_args = sys_args[1:]
    passed_args_dict = {}
    for i in range(int(len(sys_args) / 2)):
        conf_name = sys_args[2 * i].replace('-', '')
        passed_args_dict[conf_name] = sys_args[2 * i + 1]
    return passed_args_dict


def get_parser_opt_args():
    parser_opts = parser._actions
    return [x.option_strings[0].replace('--', '') for x in parser_opts[1:]]


def get_configs_name():
    return get_parser_opt_args()


def get_default_configs():
    return vars(parser.parse_known_args()[0])


def get_host_configs():
    if 'config' in HOSTNAME_INFO:
        return HOSTNAME_INFO['config']
    else:
        return None


def separate_configs(configs: (str, dict, list)):
    """
    Function to separate the configurations into categories.
    The category is defined by the first word preceding "_" in the argument.
    Ex: dataset_dir is member of the dataset category.
    For configuration without category (no _), simply add to the dict.

    Args:
        configs (str, dict, list): filename(s) of config file or dict(s) containing the configuration values.
    Returns:
        dict: a 2 level dict where args are accessed as dict[category][conf]
    """
    is_list = isinstance(configs, list)
    if not is_list:
        configs = [configs]
    is_str = isinstance(configs[0], str)
    is_dict = isinstance(configs[0], dict)
    if not is_str ^ is_dict:
        raise Exception('The input must either be a str or a dict')

    separated_configs_arr = []

    for conf in configs:
        if is_str:
            configs_dict = load_configs(conf)
        elif is_dict:
            configs_dict = conf
        separated_configs = {}

        for conf_name in configs_dict:
            if "_" in conf_name:
                config_category, config_name = separate_config_name(conf_name)
                if config_category not in separated_configs:
                    separated_configs[config_category] = {}
                separated_configs[config_category][config_name] = configs_dict[conf_name]
            else:
                separated_configs[conf_name] = configs_dict[conf_name]

        separated_configs_arr.append(separated_configs)

    if is_str:
        save_configs(separated_configs_arr, configs)
    if is_dict:
        if is_list:
            return separated_configs_arr
        else:
            return separated_configs_arr[0]


def separate_config_name(configs: (str, list)):
    is_list = isinstance(configs, list)
    if not is_list:
        configs = [configs]
    separated_config_names = []
    for config_name in configs:
        if "_" in config_name:
            separator_ind = config_name.index('_')
            config_category = config_name[:separator_ind]
            config_name = config_name[separator_ind + 1:]
            separated_config_names.append((config_category, config_name))
        else:
            separated_config_names.append(config_name)
    if is_list:
        return separated_config_names
    else:
        return separated_config_names[0]


def join_configs_categories(config_in):
    # make a hard copy
    config = dict(config_in)

    # ensure that all keyvalues are at most 1 depth dictionaries
    two_level_deep = False
    for key1 in config:
        keyvalue1 = config[key1]
        if isinstance(keyvalue1, dict):
            two_level_deep = True
            config[key1] = join_configs_categories(keyvalue1)

    if two_level_deep:
        joined_dict = {}
        for key1 in config:
            if isinstance(config[key1], dict):
                for key2 in config[key1]:
                    joined_dict[key1 + '_' + key2] = config[key1][key2]
            else:
                joined_dict[key1] = config[key1]
    else:
        joined_dict = config
    return joined_dict


def build_config(input_config):
    """
    Method to get the configuration object
    :param input_config: filename or dictionary with keys matching the parser options.
    :return: object where configuration categories are subclasses. Ex: config.dataset.dir returns the dataset directory.
    """
    if isinstance(input_config, str):
        with open(os.path.join(configs_dir, input_config)) as fid:
            input_config_dict = json.load(fid)
    elif isinstance(input_config, dict):
        input_config_dict = input_config
    else:
        raise Exception('Input config format not recognized')

    # ensure that the input_config is well defined.
    configs_name = get_configs_name()
    input_config_joined = join_configs_categories(input_config_dict)
    missing_configurations = list(set(input_config_joined.keys()).symmetric_difference(configs_name))
    if missing_configurations:
        raise Exception('The configuration: {} is missing or not defined.'.format(".".join(missing_configurations)))

    default_config_dict = get_default_configs()
    for name in configs_name:
        if input_config_joined[name] is None and input_config_joined[name] != default_config_dict[name]:
            raise Exception('Configuration {} has no value'.format(name))

    separated_config = separate_configs(input_config_joined)
    config_obj = create_config_obj(separated_config)
    config_obj.build_absolute_paths()
    return config_obj


def add_missing_defaults(configs: (str, dict, list)):
    is_list = isinstance(configs, list)
    if not is_list:
        configs = [configs]
    is_str = isinstance(configs[0], str)
    is_dict = isinstance(configs[0], dict)
    if not is_str ^ is_dict:
        raise Exception('The input must either be a str or a dict')

    updated_configs_arr = []
    for conf in configs:
        if is_str:
            filename = conf
            configs_dict = load_configs(filename)
        elif is_dict:
            configs_dict = conf

        default_configs = get_default_configs()
        updated_configs = join_configs_categories(configs_dict)
        for default_conf_name in default_configs:
            if default_conf_name not in updated_configs:
                # test if the new data is json serializable.
                json.dumps(default_configs[default_conf_name])
                updated_configs[default_conf_name] = default_configs[default_conf_name]
        updated_configs_arr.append(updated_configs)

    if is_str:
        save_configs(updated_configs_arr, configs)
    if is_dict:
        if is_list:
            return updated_configs_arr
        else:
            return updated_configs_arr[0]


def replace_configs(configs: (str, dict, list), new_configurations_dict):
    is_list = isinstance(configs, list)
    if not is_list:
        configs = [configs]
    is_str = isinstance(configs[0], str)
    is_dict = isinstance(configs[0], dict)
    if not is_str ^ is_dict:
        raise Exception('The input must either be a str or a dict')

    updated_configs_arr = []
    for conf in configs:
        if is_str:
            configs_dict = load_configs(conf)
        elif is_dict:
            configs_dict = conf

        updated_configs = join_configs_categories(configs_dict)
        for conf_name in new_configurations_dict:
            if conf_name in updated_configs:
                # test if the new data is json serializable.
                json.dumps(new_configurations_dict[conf_name])
                updated_configs[conf_name] = new_configurations_dict[conf_name]
            else:
                raise Exception('{} does not exist.'.format(conf_name))

        updated_configs_arr.append(updated_configs)

    if is_str:
        save_configs(updated_configs_arr, configs)
    if is_dict:
        if is_list:
            return updated_configs_arr
        else:
            return updated_configs_arr[0]


def update_configs(configs_filenames, new_configurations=None):
    """
    Function to update a configuration file in the configs directory.
    :param configs_filenames: configuration filenames
    :param new_configurations: determine which configuration will be overwritten by the default value
    :return: updated 2-level dictionary where 1st level keys are configuration categories
    """
    if not isinstance(configs_filenames, list):
        configs_filenames = [configs_filenames]

    for filename in configs_filenames:
        updated_configs = load_configs(filename)
        # add missing default configurations
        updated_configs = add_missing_defaults(updated_configs)

        # overwrite the host-specific configurations
        updated_configs = overwrite_with_host(updated_configs)

        # overwrite the new_configurations
        if new_configurations:
            updated_configs = replace_configs(updated_configs, new_configurations)

        # remove deprecated configurations
        deleted_options = ['help', 'output_project_dir']
        for opt in deleted_options:
            if opt in updated_configs:
                del updated_configs[opt]

        # reorder the configuration names in the order of the parser
        updated_configs = reorder_configs(updated_configs)

        # build the configs object as a check and save
        updated_configs = separate_configs(updated_configs)
        create_config_obj(updated_configs)
        save_configs(updated_configs, filename)


def update_configs_all(new_configuration=None):
    configs_filenames = [file for file in os.listdir(configs_dir) if file.endswith('.json')]
    configs_filenames.pop(configs_filenames.index("host_config.json"))
    update_configs(configs_filenames, new_configuration)


def specific_update(configs_filenames):
    if not isinstance(configs_filenames, list):
        configs_filenames = [configs_filenames]

    for filename in configs_filenames:
        with open(os.path.join(configs_dir, filename)) as fid:
            input_config_dict = json.load(fid)
        config_obj = create_config_obj(input_config_dict)
        new_configs = {}

        # rewrite the default eval model path
        new_configs['eval_model_name'] = 'weights/ssd300_' + config_obj.dataset.name + '_Final.pth'

        # rewrite the dataset dir from the dataset.name
        TreeseriesID, SynthesisID = config_obj.dataset.name.split('_')
        if TreeseriesID in ['Tree28', 'Tree29']:
            sabt_type = 'sabt14b'
        elif TreeseriesID in ['Tree30']:
            sabt_type = 'sabt16'
        else:
            raise Exception('The sabt type could not be determined.')

        new_configs['dataset_dir'] = "Dendritic Arbor/{}/{}/Synthetic Images/{}".format(sabt_type, TreeseriesID,
                                                                                        SynthesisID)
        replace_configs(filename, new_configs)


def specific_update_all():
    configs_filenames = [file for file in os.listdir(configs_dir) if file.endswith('.json')]
    configs_filenames.pop(configs_filenames.index("host_config.json"))
    specific_update(configs_filenames)


def overwrite_with_host(configs: (str, dict)):
    is_str = isinstance(configs, str)
    is_dict = isinstance(configs, dict)
    if 'config' in HOSTNAME_INFO:
        if is_str:
            replace_configs(configs, HOSTNAME_INFO['config'])
        elif is_dict:
            return replace_configs(configs, HOSTNAME_INFO['config'])
    else:
        return configs


def overwrite_with_host_all():
    if 'config' in HOSTNAME_INFO:
        update_configs_all(HOSTNAME_INFO['config'])


def reformat_json(json_dump):
    """
    Reformats a json dump to put all arrays on one line.
    :param json_dump: any output of json.dumps(dict)
    :return: a new formatted dump
    """
    # put arrays on one line
    brace_counter = 0
    array_begin_ind = 0
    array_end_ind = 0
    new_dump = ''
    for i in range(len(json_dump)):
        str = json_dump[i]
        brace_counter_prev = brace_counter
        brace_counter += 1 if str == '[' else -1 if str == ']' else 0

        if brace_counter_prev == 0 and brace_counter == 1:
            array_begin_ind = i
        if brace_counter_prev == 1 and brace_counter == 0:
            array_end_ind = i
        if brace_counter_prev == 0 and brace_counter == 0:
            new_dump += str

        if array_end_ind > array_begin_ind and brace_counter == 0:
            array_str = json_dump[array_begin_ind:array_end_ind + 1]
            array_str = re.sub(r'\s+', '', array_str)
            array_str.replace(',', ', ')
            new_dump += array_str

            array_begin_ind = 0
            array_end_ind = 0
    return new_dump


def reorder_configs(config_dict):
    configuration_names = get_configs_name()
    ordered_dict = OrderedDict()

    for conf_name in configuration_names:
        ordered_dict[conf_name] = config_dict[conf_name]

    return ordered_dict


def load_configs(configs_filename):
    with open(os.path.join(configs_dir, configs_filename)) as fid:
        input_config_dict = json.load(fid)
    return input_config_dict


def save_configs(config_dicts, configs_filenames):
    if not isinstance(config_dicts, list):
        config_dicts = [config_dicts]
        configs_filenames = [configs_filenames]

    for config_dict, configs_filename in zip(config_dicts, configs_filenames):
        configs_filepath = os.path.join(configs_dir, configs_filename)
        with open(configs_filepath, 'w') as fid:
            separated_config_dict = separate_configs(config_dict)
            fid.write(reformat_json(json.dumps(separated_config_dict, indent=4)))


def rename_config(configs_filenames, old_conf_name, new_conf_name):
    if not isinstance(configs_filenames, list):
        configs_filenames = [configs_filenames]

    for filename in configs_filenames:
        configs_path = os.path.join(configs_dir, filename)
        with open(configs_path, 'r') as fid:
            configs_dict = json.load(fid)

        updated_configs = OrderedDict(join_configs_categories(configs_dict))
        updated_configs = OrderedDict(
            (new_conf_name if k == old_conf_name else k, v) for k, v in updated_configs.items())
        save_configs(updated_configs, filename)


def rename_config_all(old_conf_name, new_conf_name):
    configs_filenames = [file for file in os.listdir(configs_dir) if file.endswith('.json')]
    configs_filenames.pop(configs_filenames.index("host_config.json"))
    rename_config(configs_filenames, old_conf_name, new_conf_name)


if __name__ == "__main__":
    # # Example usages of the methods above.
    #
    # # Reset configurations to default.
    # updated_conf_names = ['dataset_bounding_boxes_dir', 'dataset_images_dir', 'output_weights_dir',
    #                       'output_detections_dir']
    # updated_conf_names = ['eval_model_name']
    # default_conf = get_default_configs()
    # new_conf_dict = {key: default_conf[key] for key in default_conf if key in updated_conf_names}
    # update_configs_all(new_conf_dict)
    #
    # # Build the configuration object.
    # obj = build_config('Tree28_synthesis1_config.json')
    #
    # # Perform updates specific to each file. Check specific_update()
    # specific_update_all()
    #
    # # Rename a configuration
    # rename_config('Tree28_synthesis1_config.json','output_weights_dir','output_weights_folder')
    # rename_config_all('output_weights_dir', 'output_weights_folder')
    # rename_config_all('output_weights_folder','output_weights_dir')

    # Update a configuration file
    # filename = 'Tree28_synthesis1_config.json'
    # update_configs(filename)

    # Update all configuration files.
    # update_configs_all()

    # parse
    args = parser.parse_args()

    print(args)
    configs_parsed_dict = vars(args)
    configs_separated = separate_configs(configs_parsed_dict)

    # save configs
    if not os.path.isdir(configs_dir):
        os.makedirs(configs_dir)
    configs_filename = configs_parsed_dict['dataset_name'] + '_config.json'
    configs_path = os.path.join(configs_dir, configs_filename)
    if not os.path.isfile(configs_path):
        save_configs(configs_separated, configs_filename)
        overwrite_with_host(configs_filename)
    else:
        raise Exception('The configuration file: {} already exists.'.format(configs_path))

    print("Created a configuration file with the following arguments:")
    print(reformat_json(json.dumps(configs_separated)))
