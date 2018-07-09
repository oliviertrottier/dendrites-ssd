"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import csv
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import re

FOREST_SIZE=25

class ObjectTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __call__(self, objects, properties_name, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # Scale the height and width of the bounding boxes.
        # The bounding box properties of the dataset are given in the format: properties_format.
        # Change the box properties to the format: (xmin, ymin, xmax, ymax, class).
        output_properties = ['xmin', 'ymin', 'xmax', 'ymax', 'class']
        new_format = tuple([properties_name.index(x) for x in output_properties])

        # Since the pixel index starts at 1, subtract 1 to map (1,width/height) to (0,1)
        new_objects = objects[:, new_format]
        dividor = np.array([width, height, width, height]).reshape(-1,4)
        new_objects[:, :4] = np.divide(new_objects[:, :4] - 1, dividor)

        return new_objects


class TreeDataset(data.Dataset):
    """Tree Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, config, forest_size=FOREST_SIZE,
                 transform=None, object_transform=ObjectTransform()):
        self.name = config.name
        self.tree_series = self.name.split('_')[0]
        self.image_filename_format = self.tree_series + '_{}_{}'

        self.root = config.dir
        self.images_dir = osp.join(self.root, config.images_dir)
        self.objects_dir = osp.join(self.root, config.bounding_boxes_dir)
        self.object_properties_name = config.object_properties

        self.transform = transform
        self.object_transform = object_transform

        self.forest_size=forest_size

        # Get all .jpg filenames in the image path.
        self.filenames = list()
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.filenames.append(osp.splitext(file)[0])

        # Sort the filenames in ascending tree id.
        self.filenames.sort(key=self.filename_to_index)

    def __getitem__(self, index):
        # Import the image.
        filename = self.filenames[index]
        img = cv2.imread(osp.join(self.images_dir, filename + '.jpg'))
        height, width, channels = img.shape

        # Get the bounding box limits and class of objects in the image.
        objects_properties = self.get_raw_gt(index)

        # Transform the objects objects_properties to numpy arrays.
        objects_properties = np.array(objects_properties, dtype=float)
        # objects_box_limits = objects_properties[:,:4]
        # objects_box_limits = np.array(objects_box_limits)
        # objects_class = np.array(objects_class, dtype=float).reshape(-1,1)
        # objects_properties = np.hstack((objects_box_limits,objects_class))

        # Transform the object's box limits.
        if self.object_transform is not None:
            objects_properties = self.object_transform(objects_properties, self.object_properties_name, width, height)

        # Transform the image
        if self.transform is not None:
            img, boxes, labels = self.transform(img, objects_properties[:, :4], objects_properties[:, 4])
            targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # Transform to torch tensor and permute dimensions to bring color channels first.
        torch_img = torch.from_numpy(img).permute(2, 0, 1)
        return torch_img, targets

    def __len__(self):
        return len(self.filenames)

    def get_raw_gt(self, i):
        # Get the raw ground truth objects in image i.
        filename = self.filenames[i]
        objects_properties = list()
        with open(osp.join(self.objects_dir, filename + '.csv'), newline='') as csvfile:
            csv_content = csv.reader(csvfile, delimiter=',')
            # Skip header.
            next(csv_content, None)
            for row in csv_content:
                objects_properties.append([int(x) for x in row])
        return objects_properties

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        # img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        # gt = self.target_transform(anno, 1, 1)
        # return img_id[1], gt
        pass

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        # return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
        pass

    def index_to_filename(self, index):
        return self.image_filename_format.format((int(index/self.forest_size) + 1, index))

    def filename_to_index(self,filenames_input):
        if not isinstance(filenames_input,list):
            filenames = [filenames_input]
        else:
            filenames = filenames_input
        indices=list()
        for filename in filenames:
            #groups = parse(IMAGE_FILENAME_FORMAT, filename)
            groups = re.findall('(?<=_)\d+', filename)
            indices.append(int(groups[1]))

        if not isinstance(filenames_input,list):
            indices = indices[0]
        return indices
