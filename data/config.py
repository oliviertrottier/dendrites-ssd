# config.py
import sys
import os
import json
import re
from argparse import ArgumentParser

# gets home dir cross platform
# HOME = os.path.expanduser("~")
HOME = os.path.dirname(os.path.realpath(sys.argv[0]))

script_dir = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'
configs_dir = os.path.join(script_dir, 'configs/')

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
                    help='Root subdirectory folder where bounding boxes properties are saved')

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
parser.add_argument('--train_resume', type=str,
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

# eval
parser.add_argument('--eval_trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--eval_overwrite_all_detections', default=False, type=bool,
                    help='Overwrite all_detections file')
parser.add_argument('--eval_confidence_threshold', default=0.01, type=float,
                    help='Discard detected boxes below confidence threshold')
parser.add_argument('--eval_top_k', default=50, type=int,
                    help='Restrict the number of predictions per image')
parser.add_argument('--eval_cuda', default=True, type=bool,
                    help='Use cuda to evaluate the model')

# criterion
parser.add_argument('--criterion_train', type=str, default='multibox')

# output
parser.add_argument('--output_weights_folder', type=str, default='weights/',
                    help='Directory for saving model training checkpoints')
parser.add_argument('--output_detections_folder', type=str, default='detections/',
                    help='Root subdirectory folder where detections are output')

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
    'dataset_root': 'root/test',
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
    def __init__(self, root, name, num_classes, classes_name, images_dir, object_properties, bounding_boxes_dir):
        self.root = root
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
    def __init__(self, trained_model, overwrite_all_detections, confidence_threshold, top_k, cuda):
        self.trained_model = trained_model
        self.overwrite_all_detections = overwrite_all_detections
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.cuda = cuda


class criterion:
    def __init__(self, train):
        self.train = train


class output:
    def __init__(self, weights_folder, detections_folder):
        self.weights_folder = weights_folder
        self.detections_folder = detections_folder


class configs:
    def __init__(self, dataset, dataloader, train, model, eval, criterion, output):
        self.dataset = dataset
        self.dataloader = dataloader
        self.train = train
        self.model = model
        self.eval = eval
        self.criterion = criterion
        self.output = output


def get_parser_opt_args():
    parser_opts = parser._actions
    return [x.option_strings[0].replace('--', '') for x in parser_opts[1:]]


def get_configs_name():
    return get_parser_opt_args()


def get_default_configs():
    return vars(parser.parse_known_args()[0])


def separate_configs_categories(config_dict):
    """
    Function to separate the configurations into categories.
    The category is defined by the first word preceding _ in the argument.
    Ex: dataset_root is member of the dataset category.
    For configuration without category (no _), simply add to the dict.
    :param config_dict: dict containing the configuration values.
    :return: 2 level dict where args are accessed as dict[category][conf]
    """
    configs_separated = {}
    for conf in config_dict:
        if "_" in conf:
            separator_ind = conf.index('_')
            config_category = conf[:separator_ind]
            config_name = conf[separator_ind + 1:]
            if config_category not in configs_separated:
                configs_separated[config_category] = {}
            configs_separated[config_category][config_name] = config_dict[conf]
        else:
            configs_separated[conf] = config_dict[conf]
    return configs_separated


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


def build_config(config_dict):
    dataset_dict = config_dict['dataset']
    root = dataset_dict['root']
    name = dataset_dict['name']
    num_classes = dataset_dict['num_classes']
    classes_name = dataset_dict['classes_name']
    images_dir = dataset_dict['images_dir']
    object_properties = dataset_dict['object_properties']
    bounding_boxes_dir = dataset_dict['bounding_boxes_dir']
    dataset_conf = dataset(root, name, num_classes, classes_name, images_dir, object_properties, bounding_boxes_dir)

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
    trained_model = eval_dict['trained_model']
    overwrite_all_detections = eval_dict['overwrite_all_detections']
    confidence_threshold = eval_dict['confidence_threshold']
    top_k = eval_dict['top_k']
    cuda = eval_dict['cuda']
    eval_conf = eval(trained_model, overwrite_all_detections, confidence_threshold, top_k, cuda)

    criterion_dict = config_dict['criterion']
    criterion_conf = criterion(criterion_dict['train'])

    output_dict = config_dict['output']
    weights_folder = output_dict['weights_folder']
    detections_folder = output_dict['detections_folder']
    output_conf = output(weights_folder, detections_folder)

    configs_obj = configs(dataset_conf, dataloader_conf, train_conf, model_conf, eval_conf, criterion_conf, output_conf)

    # check that the config object has all the defined configurations
    for category in config_dict:
        for conf in config_dict[category]:
            getattr(getattr(configs_obj, category), conf)

    return configs_obj


def get_config(input_config):
    """
    Method to get the configuration object
    :param input_config: filename or dictionary with keys matching the parser options.
    :return: object where configuration categories are subclasses. Ex: config.dataset.root returns the dataset root.
    """
    if isinstance(input_config, str):
        with open(os.path.join(configs_dir, input_config)) as fid:
            input_config_dict = json.load(fid)
    elif isinstance(input_config, dict):
        input_config_dict = input_config
    else:
        raise Exception('Input config format not recognized')

    # ensure that the input_config are well defined.
    configs_name = get_configs_name()
    input_config_joined = join_configs_categories(input_config_dict)
    missing_configurations = list(set(input_config_joined.keys()).symmetric_difference(configs_name))
    if missing_configurations:
        raise Exception('The configuration: {} is missing or not defined.'.format(".".join(missing_configurations)))

    default_config_dict = get_default_configs()
    for name in configs_name:
        if input_config_joined[name] is None and input_config_joined[name] != default_config_dict[name]:
            raise Exception('Configuration {} has no value'.format(name))

    separated_config = separate_configs_categories(input_config_joined)
    config_obj = build_config(separated_config)

    return config_obj


def update_configs_file(configs_filename, overwritten_configuration=None):
    """
    Function to update a configuration file in the configs directory.
    :param configs_filename: configuration filename
    :param overwritten_configuration: determine which configuration will be overwritten by the default value
    :return: updated 2-level dictionary where 1st level keys are configuration categories
    """
    configs_path = os.path.join(configs_dir, configs_filename)
    with open(configs_path, 'r') as fid:
        configs_dict = json.load(fid)

    # add missing defaults value
    if not isinstance(overwritten_configuration, list):
        overwritten_configuration = list(overwritten_configuration)
    default_configs = get_default_configs()
    updated_configs = join_configs_categories(configs_dict)
    for default_conf_name in default_configs:
        if default_conf_name not in updated_configs or default_conf_name in overwritten_configuration:
            updated_configs[default_conf_name] = default_configs[default_conf_name]

    # remove deprecated conf
    if 'help' in updated_configs:
        del updated_configs['help']

    # test the new configs.
    updated_configs = separate_configs_categories(updated_configs)
    build_config(updated_configs)

    save_configs(updated_configs, configs_path)
    return updated_configs


def update_all_configs():
    configs_files = os.listdir(configs_dir)
    for file in configs_files:
        update_configs_file(file)


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


def save_configs(config_dict, filepath):
    with open(filepath, 'w') as fid:
        fid.write(reformat_json(json.dumps(config_dict, indent=4)))


if __name__ == "__main__":
    # parse
    args = parser.parse_args()

    print(args)
    configs_parsed_dict = vars(args)
    configs_separated = separate_configs_categories(configs_parsed_dict)

    # save configs
    if not os.path.isdir(configs_dir):
        os.makedirs(configs_dir)
    configs_path = os.path.join(configs_dir, configs_parsed_dict['dataset_name'] + '_config.json')
    if not os.path.isfile(configs_path):
        save_configs(configs_separated, configs_path)
    else:
        raise Exception('The configuration file: {} already exists.'.format(configs_path))

    print("Created a configuration file with the following arguments:")
    print(reformat_json(json.dumps(configs_separated)))
