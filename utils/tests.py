# Series of tests to check preprocessing routines
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils.augmentations import ToAbsoluteCoords

ToAbsoluteCoordsTransform = ToAbsoluteCoords()


def plot_image(image, boxes=None):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add boxes to the image.
    if boxes is not None:
        boxes2 = boxes.copy()
        N_boxes = boxes2.shape[0]
        cmap = plt.get_cmap('hsv')
        N_classes = np.unique(boxes2[:, 4]).size

        # Convert to integer coordinates if in % coordinates.
        if boxes2[0, 0] < 1:
            _, boxes2, _ = ToAbsoluteCoordsTransform(image, boxes2)

        # Plot each box using the class to define the color.
        for i in range(N_boxes):
            bottom_left_corner = [boxes2[i, 0], boxes2[i, 1]]
            width = boxes2[i, 2] - boxes2[i, 0]
            height = boxes2[i, 3] - boxes2[i, 1]
            object_class = boxes2[i, 4]
            color_triple = cmap(object_class / N_classes)

            # Plot the bounding box.
            rect = patches.Rectangle(tuple(bottom_left_corner), width, height, linewidth=1,
                                     edgecolor=color_triple, facecolor='none')
            # Plot the center of the box.
            plt.scatter(bottom_left_corner[0]+width/2, bottom_left_corner[1]+height/2, c=color_triple,s=1)

            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.show()
