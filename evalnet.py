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
from data import VOC_ROOT, ObjectTransform, TreeDataset, BaseTransform
from data import tree_synth0_config, tree_synth1_config, tree_synth2_config
import torch.utils.data as data

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

from layers.box_utils import jaccard, intersect

labelmap = tree_synth0_config['classes_name']

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--dataset_root',
                    help='Location of dataset root directory')
parser.add_argument('--dataset',
                    help='Name of the dataset')
parser.add_argument('--detections_folder', default='Detections/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=50, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()
if args.detections_folder == 'Detections/':
    args.detections_folder = os.path.join(args.dataset_root, args.detections_folder)

if not os.path.exists(args.detections_folder):
    os.mkdir(args.detections_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Determine the configuration.
if args.dataset in ['Tree28_synthesis1', 'Tree29_synthesis1']:
    dataset_config = tree_synth0_config
elif args.dataset in ['Tree28_synthesis2', 'Tree29_synthesis2']:
    dataset_config = tree_synth1_config
elif args.dataset in ['Tree28_synthesis3', 'Tree29_synthesis3']:
    dataset_config = tree_synth2_config

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'
all_detections_filename = 'All_Detections.pkl'
ALL_DETECTIONS_FILENAME = os.path.join(args.detections_folder, all_detections_filename)
DETECTION_STATISTICS_FILENAME = os.path.join(args.detections_folder, 'detections_statistics.txt')


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


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                               annopath,
                               imagesetfile,
                               classname,
                               [ovthresh],
                               [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def detect_objects(detections_folder, net, config, dataset):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_detections[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_detections = [[[] for _ in range(config['num_classes'])]
                      for _ in range(num_images)]

    detections_column_names = ['xmin', 'xmax', 'ymin', 'ymax', 'class', 'class_score']

    for i in range(num_images):
        # Start timer.
        timer = Timer()
        timer.tic()
        # Get image.
        im, box_limits_gt = dataset[i]
        h, w = im.size()[1:]
        x = Variable(im.unsqueeze(0))
        detections_csv_output = np.array([], dtype=np.float).reshape(0, 6)

        if args.cuda:
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
            class_type = j * np.ones(scores.shape)
            box_limits = np.round(boxes.cpu().numpy()[:, (0, 2, 1, 3)]) + 1
            all_detections[i][j] = np.hstack((box_limits, scores)).astype(np.float32, copy=False)
            cls_dets = np.hstack((box_limits, class_type, scores)).astype(np.float32, copy=False)
            detections_csv_output = np.concatenate((detections_csv_output, cls_dets), 0)

        # Save image detections in .csv file
        # image_detections = np.concatenate(np.asarray(objects[1:]), 0)
        box_limits = detections_csv_output[:, 0:4].astype('uint16')
        # jac=jaccard(torch.Tensor(box_limits[:,(0,2,1,3)]), torch.Tensor(box_limits_gt[:,:4]))

        class_types = detections_csv_output[:, 4].astype('uint8')
        class_scores = detections_csv_output[:, 5]
        # np.savetxt(os.path.join(detections_folder, dataset.filenames[i] + '.csv'), image_detections, delimiter=",")
        with open(os.path.join(detections_folder, dataset.filenames[i] + '.csv'), 'w', newline='') as csvfile:
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
    with open(ALL_DETECTIONS_FILENAME, 'wb') as f:
        pickle.dump(all_detections, f, pickle.HIGHEST_PROTOCOL)
        print("Saved all detections in {}".format(ALL_DETECTIONS_FILENAME))


def evaluate_detections(dataset):
    # Load all the detections.
    with open(ALL_DETECTIONS_FILENAME, 'rb') as file:
        all_detections = pickle.load(file)
    num_images = len(dataset)
    num_classes = dataset.num_classes
    classes_name = dataset.classes_name

    # Loop over images.
    # For each image, calculate
    # 1) the highest jaccard index for each ground truth.
    # 2) true positives and false positives/negatives for each class.
    true_pos = np.nan * np.zeros((num_images, num_classes))
    false_pos = np.nan * np.zeros((num_images, num_classes))
    false_neg = np.nan * np.zeros((num_images, num_classes))
    truepos_jaccard_mean = np.nan * np.zeros((num_images, num_classes))
    min_jaccard_overlap = 0.5
    for i in range(num_images):
        boxes_limits_image_gt = np.array(dataset.get_raw_gt(i))
        boxes_class_image = boxes_limits_image_gt[:, -1]
        for j in range(1, num_classes):
            # TODO: change j-1 to j after removing class 0 in dataset.
            boxes_limits_gt = boxes_limits_image_gt[boxes_class_image == (j - 1), :4]
            image_detections = all_detections[i][j]
            N_detections = len(image_detections)
            N_gt = boxes_limits_gt.shape[0]
            if N_detections == 0:
                true_pos[i, j] = 0
                false_neg[i, j] = N_gt
                false_pos[i, j] = 0
                truepos_jaccard_mean[i, j] = 0
                continue

            # Get the box limits and confidence.
            boxes_limits_detections = image_detections[:, :4]
            detections_conf = image_detections[:, 4]

            # Calculate the jaccard matrix.
            boxes_limits_gt_tensor = torch.Tensor(boxes_limits_gt)
            boxes_limits_detections_tensor = torch.Tensor(boxes_limits_detections)

            # Permute columns to satisfy the input format of jaccard.
            boxes_limits_gt_tensor = boxes_limits_gt_tensor[:, (0, 2, 1, 3)]
            boxes_limits_detections_tensor = boxes_limits_detections_tensor[:, (0, 2, 1, 3)]

            #intersect(boxes_limits_gt_tensor[0,:].unsqueeze(0),boxes_limits_detections_tensor[13,:].unsqueeze(0))
            jaccard_mat = jaccard(boxes_limits_gt_tensor, boxes_limits_detections_tensor)

            # Match each grount truth to a detection, if overlap > overlap_min.
            best_overlap_jaccard, best_overlap_index = jaccard_mat.max(0)
            gt_match_ind = np.zeros((N_gt, 1))*np.nan
            gt_match_conf = np.zeros((N_gt, 1))

            for k in range(N_detections):
                best_gt_ind = best_overlap_index[k]
                if detections_conf[k] > gt_match_conf[best_gt_ind] and best_overlap_jaccard[k] > min_jaccard_overlap:
                    gt_match_ind[best_gt_ind] = k
                    gt_match_conf[best_gt_ind] = detections_conf[k]

            # Remove nans, which correspond to unmatched gt boxes.
            gt_match_ind = gt_match_ind[~np.isnan(gt_match_ind)].astype(np.int16)

            # Calculate the true positives and false positives/negatives.
            true_pos[i, j] = gt_match_ind.shape[0]
            false_neg[i, j] = N_gt - true_pos[i][j]
            false_pos[i, j] = len([x for x in range(N_detections) if not x in gt_match_ind])
            truepos_jaccard_mean[i, j] = np.mean(best_overlap_jaccard.numpy()[gt_match_ind])

    statistics_dict=dict(zip(classes_name, [{}]*len(classes_name)))
    for j in range(1, num_classes):
        class_dict = {}
        class_dict['N_groundtruths'] = np.sum(true_pos[:, j] + false_neg[:, j])
        class_dict['N_detections'] = np.sum(true_pos[:, j] + false_pos[:, j])
        class_dict['True Positives'] = np.sum(true_pos[:, j])
        class_dict['False Positives'] = np.sum(false_pos[:, j])
        class_dict['False Negatives'] = np.sum(false_neg[:, j])
        class_dict['Precision'] = class_dict['True Positives'] / (class_dict['True Positives'] + class_dict['False Positives'])
        class_dict['Recall'] = class_dict['True Positives'] / (class_dict['True Positives'] + class_dict['False Negatives'])
        class_dict['Jaccard_TruePos_Average'] = np.mean(truepos_jaccard_mean[:, j])

        total_false = false_pos[:, j] + false_neg[:, j]
        worst_image_id = np.argmax(total_false)

        class_dict['Highest_error_image'] = dataset.filenames[worst_image_id]
        class_dict['Highest_error_image_errors'] = total_false[worst_image_id]

        statistics_dict[classes_name[j]] = class_dict
    with open(DETECTION_STATISTICS_FILENAME,'w') as file:
        json.dump(statistics_dict, file, sort_keys=False, indent=2)


if __name__ == '__main__':
    # Load neural net.
    net = build_ssd('test', dataset_config)
    if not args.cuda:
        net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # Load dataset.
    dataset = TreeDataset(root=args.dataset_root, name=args.dataset,
                          transform=BaseTransform(300, dataset_config['pixel_means']))

    # Detect objects.
    if not os.path.isfile(ALL_DETECTIONS_FILENAME):
        detect_objects(args.detections_folder, net, dataset_config, dataset)
    else:
        print("An All_Detections file has been detected. Skipping object detections.")

    # Add useful properties to dataset.
    dataset.num_classes = dataset_config['num_classes']
    dataset.classes_name = dataset_config['classes_name']

    # Evaluate detections
    evaluate_detections(dataset)
