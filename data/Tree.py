import os, sys
import csv
import os.path as osp

import torch
import torch.utils.data as data
import cv2
import numpy as np
import re
from .config import dataset
from utils.augmentations import ToPercentCoords

# Pattern used to assign ID number to an image. If the pattern is not found, the alphabetical order is used instead.
FILENAME_ID_PATTERN = '\d+'


class TreeDataset(data.Dataset):
    """Tree Detection Dataset Object

    Arguments:
        config (object): dataset config object created from config.py
    """

    def __init__(self, config: dataset, transform=None):
        self.name = config.name
        self.tree_series = self.name.split('_')[0]

        self.root = config.dir
        self.images_dir = osp.join(self.root, config.images_dir)
        self.objects_dir = osp.join(self.root, config.bounding_boxes_dir)

        self.num_classes = config.num_classes
        self.classes_name = config.classes_name
        self.object_properties_name = config.object_properties

        self.transform = transform

        # Get all .jpg filenames in the image directory
        self.filenames = list()
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.filenames.append(osp.splitext(file)[0])

        # Sort the filenames in ascending tree ID
        self.filenames.sort(key=self.filename_to_ID)
        self.IDs = self.filename_to_ID(self.filenames)

    def __getitem__(self, index):
        # Import the image.
        img = self.get_image(index)

        # Get the bounding box limits and class of objects in the image
        objects_properties = self.get_gt(index)

        # Transform the objects objects_properties to numpy arrays
        objects_properties = np.array(objects_properties, dtype=float)

        # Transform the format of the objects' properties
        objects_properties = self.object_transform(objects_properties, self.object_properties_name)

        # Transform the image
        object_box_limits = objects_properties[:, :4]
        object_class = objects_properties[:, 4]
        if self.transform is not None:
            img, object_box_limits, object_class = self.transform(img, object_box_limits, object_class)

        # Transform to torch tensor and permute dimensions to bring color channels first.
        targets = np.hstack((object_box_limits, np.expand_dims(object_class, axis=1)))
        torch_img = torch.from_numpy(img).permute(2, 0, 1)
        return torch_img, targets

    def __len__(self):
        return len(self.filenames)

    def get_gt(self, index):
        # Get the ground truth objects in image.
        filename = self.filenames[index]
        filepath = osp.join(self.objects_dir, filename + '.csv')
        objects_properties = list()
        if os.path.exists(filepath):
            with open(filepath, newline='') as csvfile:
                csv_content = csv.reader(csvfile, delimiter=',')
                # Skip header.
                next(csv_content, None)
                for row in csv_content:
                    objects_properties.append([int(x) for x in row])

        # Make sure that object properties has the expected number of columns, even if empty.
        output = np.array(objects_properties, dtype=float)
        if output.ndim == 1:
            output = output.reshape(-1, len(self.object_properties_name))
        return output

    def get_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        filename = self.filenames[index]
        filepath = osp.join(self.images_dir, filename + '.jpg')
        return cv2.imread(filepath)

    def object_transform(self, objects, input_properties_name):
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
        output_properties_name = ['xmin', 'ymin', 'xmax', 'ymax', 'class']
        new_format = tuple([input_properties_name.index(x) for x in output_properties_name])

        return objects[:, new_format]

    def ID_to_filename(self, ID):
        return self.filenames[self.IDs.index(ID)]

    def filename_to_ID(self, filenames: (str, list)):
        """
        :param filenames: filename (str) of list of filenames
        :return: IDs of input filenames
        """
        is_input_str = isinstance(filenames, str)
        if is_input_str:
            filenames = [filenames]

        IDs = list()
        for filename in filenames:
            # Attempt to find ID from the filename. Use the self.filenames order if it fails.
            filename_patt_groups = re.findall(FILENAME_ID_PATTERN, filename)
            if filename_patt_groups:
                ID = int(filename_patt_groups[-1])
            else:
                ID = self.filenames.index(filename) + 1
            IDs.append(ID)

        if is_input_str:
            IDs = IDs[0]
        return IDs
