"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import TreeDataset, BaseTransform
from data.config import build_config
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import csv
import json
import re

from layers.box_utils import jaccard, intersect

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--config', type=str,
                    help='Name of configuration file.')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--dataset_root',
                    help='Location of dataset root directory')
parser.add_argument('--dataset',
                    help='Name of the dataset')
parser.add_argument('--detections_dir', default='detections/', type=str,
                    help='File path to save results')
parser.add_argument('--overwrite_all_detections', default=False, type=bool,
                    help='Bool to determine if all_detections should be overwritten')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=50, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')

args = parser.parse_args()

# Read the config file.
configs = build_config(args.config)
TREEDATASET_PATTERN = re.compile('Tree.*')

if not os.path.exists(configs.output.detections_dir):
    os.mkdir(configs.output.detections_dir)

ALL_DETECTIONS_FILEPATH = os.path.join(configs.dataset.dir, 'all_detections.pkl')
DETECTION_STATISTICS_FILEPATH = os.path.join(configs.dataset.dir, 'detections_statistics.json')


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def detect_objects(config, net, dataset):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_detections[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    detections_dir = config.output.detections_dir
    all_detections = [[[] for _ in range(config.model.num_classes)]
                      for _ in range(num_images)]

    detections_column_names = list(config.dataset.object_properties)
    detections_column_names.append('class_score')
    for i in range(num_images):
        # Start timer.
        timer = Timer()
        timer.tic()
        # Get image.
        im, _ = dataset[i]
        h, w = im.size()[1:]
        x = Variable(im.unsqueeze(0))
        detections_csv_output = np.array([], dtype=np.float).reshape(0, 6)

        if configs.eval.cuda:
            x = x.cuda()

        # Get neural net detections.
        detections = net(x).data

        # Loop over classes. Skip j = 0 (background class).
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.nelement() == 0:
                continue
            # Scale the boxes dimensions with the image height/width.
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()[:, np.newaxis]

            # Save the class type of the box.
            class_type = (j - 1) * np.ones(scores.shape)
            box_limits = np.round(boxes.cpu().numpy()[:, (0, 2, 1, 3)]) + 1
            cls_dets = np.hstack((box_limits, class_type, scores)).astype(np.float32, copy=False)
            detections_csv_output = np.concatenate((detections_csv_output, cls_dets), 0)

            # Append to all_detections
            all_detections[i][j] = np.hstack((box_limits, scores)).astype(np.float32, copy=False)

        # Cache image detections in .csv format
        # image_detections = np.concatenate(np.asarray(objects[1:]), 0)
        box_limits = detections_csv_output[:, 0:4].astype('uint16')
        class_types = detections_csv_output[:, 4].astype('uint8')
        class_scores = detections_csv_output[:, 5]
        # np.savetxt(os.path.join(detections_dir, dataset.filenames[i] + '.csv'), image_detections, delimiter=",")
        filepath = os.path.join(detections_dir, dataset.filenames[i] + '.csv')
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=detections_column_names)
            writer.writeheader()
            for k in range(detections_csv_output.shape[0]):
                obj_dict_temp = {'xmin': box_limits[k, 0],
                                 'xmax': box_limits[k, 1],
                                 'ymin': box_limits[k, 2],
                                 'ymax': box_limits[k, 3],
                                 'class': class_types[k],
                                 'class_score': class_scores[k]}
                writer.writerow(obj_dict_temp)
        processing_time = timer.toc(average=False)
        print('{:d}/{:d}: Processed {:s} in {:.3f}s'.format(i + 1, num_images, dataset.filenames[i], processing_time))

    # Save all detections in a .pkl file.
    with open(ALL_DETECTIONS_FILEPATH, 'wb') as f:
        pickle.dump(all_detections, f, pickle.HIGHEST_PROTOCOL)
        print("Saved all detections in {}".format(ALL_DETECTIONS_FILEPATH))


def evaluate_detections(dataset, config):
    # Load all the detections.
    with open(ALL_DETECTIONS_FILEPATH, 'rb') as file:
        all_detections = pickle.load(file)
    num_images = len(dataset)
    num_classes = config.model.num_classes  # take the model num_classes, since we are omitting background
    classes_name = dataset.classes_name

    # For each image, calculate
    # 1) the highest jaccard index for each ground truth.
    # 2) true positives and false positives/negatives for each class.
    true_pos = np.nan * np.zeros((num_images, num_classes))
    false_pos = np.nan * np.zeros((num_images, num_classes))
    false_neg = np.nan * np.zeros((num_images, num_classes))
    truepos_jaccard_mean = np.nan * np.zeros((num_images, num_classes))
    min_jaccard_overlap = 0.5
    gts_exist = False

    for i in range(num_images):
        image_objects_gt = dataset.get_gt(i)
        if image_objects_gt.shape[0] == 0:
            continue
        gts_exist = True
        boxes_class_image = image_objects_gt[:, -1]
        for j in range(1, num_classes):
            boxes_limits_gt = image_objects_gt[boxes_class_image == (j - 1), :4]
            image_detections = all_detections[i][j]
            N_detections = len(image_detections)
            N_gts = boxes_limits_gt.shape[0]
            if N_detections == 0 or N_gts == 0:
                true_pos[i, j] = 0
                false_neg[i, j] = N_gts
                false_pos[i, j] = N_detections
                truepos_jaccard_mean[i, j] = 0
                continue

            # Get the box limits and confidence.
            boxes_limits_detections = image_detections[:, :4]
            detections_conf = image_detections[:, 4]

            # Calculate the jaccard overlap between detections and gt.
            boxes_limits_gt_tensor = torch.Tensor(boxes_limits_gt)
            boxes_limits_detections_tensor = torch.Tensor(boxes_limits_detections)

            # Permute columns to satisfy the input format of jaccard.
            boxes_limits_gt_tensor = boxes_limits_gt_tensor[:, (0, 2, 1, 3)]
            boxes_limits_detections_tensor = boxes_limits_detections_tensor[:, (0, 2, 1, 3)]

            # intersect(boxes_limits_gt_tensor[0,:].unsqueeze(0),boxes_limits_detections_tensor[13,:].unsqueeze(0))
            jaccard_mat = jaccard(boxes_limits_gt_tensor, boxes_limits_detections_tensor)

            # For each detection, find the best ground truth overlap.
            best_truth_jaccard, best_truth_index = jaccard_mat.max(0)

            # For each gt x, find the detection with highest confidence among all detections whose max overlap is x.
            best_detection_ind = np.zeros((N_gts, 1)) * np.nan
            best_detection_conf = np.zeros((N_gts, 1))

            for k in range(N_detections):
                best_gt_ind = best_truth_index[k]
                if best_truth_jaccard[k] > min_jaccard_overlap and detections_conf[k] > best_detection_conf[
                    best_gt_ind]:
                    best_detection_ind[best_gt_ind] = k
                    best_detection_conf[best_gt_ind] = detections_conf[k]

            # Remove nans, which correspond to unmatched gt boxes.
            best_detection_ind = best_detection_ind[~np.isnan(best_detection_ind)].astype(np.int16)

            # Calculate the true positives, false positives and false negatives.
            true_pos[i, j] = best_detection_ind.shape[0]
            false_neg[i, j] = N_gts - true_pos[i][j]
            false_pos[i, j] = len([x for x in range(N_detections) if x not in best_detection_ind])
            truepos_jaccard_mean[i, j] = np.mean(best_truth_jaccard.cpu().numpy()[best_detection_ind])

    if gts_exist:
        statistics_dict = dict(zip(classes_name, [{}] * len(classes_name)))
        for j in range(1, num_classes):
            class_dict = {}
            class_dict['N_groundtruths'] = int(np.sum(true_pos[:, j] + false_neg[:, j]))
            class_dict['N_detections'] = int(np.sum(true_pos[:, j] + false_pos[:, j]))
            class_dict['True Positives'] = int(np.sum(true_pos[:, j]))
            class_dict['False Positives'] = int(np.sum(false_pos[:, j]))
            class_dict['False Negatives'] = int(np.sum(false_neg[:, j]))
            class_dict['Precision'] = class_dict['True Positives'] / (
                    class_dict['True Positives'] + class_dict['False Positives'])
            class_dict['Recall'] = class_dict['True Positives'] / (
                    class_dict['True Positives'] + class_dict['False Negatives'])
            class_dict['Jaccard_TruePos_Average'] = np.mean(truepos_jaccard_mean[:, j])

            total_false = false_pos[:, j] + false_neg[:, j]
            worst_image_id = np.argmax(total_false)

            class_dict['Highest False Pos + Neg Image'] = dataset.filenames[worst_image_id]
            class_dict['Highest Error Image: False positives'] = int(false_pos[worst_image_id, j])
            class_dict['Highest Error Image: False negatives'] = int(false_neg[worst_image_id, j])

            statistics_dict[classes_name[j - 1]] = class_dict
        with open(DETECTION_STATISTICS_FILEPATH, 'w') as file:
            json.dump(statistics_dict, file, sort_keys=False, indent=4)
    else:
        print("No ground truths were found.")


if __name__ == '__main__':
    # Load neural net.
    net = build_ssd('test', configs.model)
    if configs.eval.cuda:
        Map_loc = lambda storage, loc: storage
    else:
        Map_loc = 'cpu'
    state_dict = torch.load(configs.eval.model_name, map_location=Map_loc)
    if 'net_state' in state_dict.keys():
        state_dict = state_dict['net_state']
    net.load_state_dict(state_dict)
    net.eval()

    if configs.eval.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # Load dataset.
    if TREEDATASET_PATTERN.match(configs.dataset.name):
        dataset = TreeDataset(configs.dataset,
                              transform=BaseTransform(configs.model.input_size, configs.model.pixel_means))
    else:
        raise Exception('The dataset is not supported.')

    # Detect objects.
    if not os.path.isfile(ALL_DETECTIONS_FILEPATH) or configs.eval.overwrite_all_detections:
        detect_objects(configs, net, dataset)
    else:
        print("{} has been detected. Skipping object detections.".format(ALL_DETECTIONS_FILEPATH))

    # Evaluate detections
    evaluate_detections(dataset, configs)
