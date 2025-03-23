import os
import cv2
import random
import numpy as np

seed = 21
np.random.seed(seed)
random.seed(seed)



n_img = 200
image_size = 512


# Modify these parameters according to the shape and number of wheat heads in your domain.
#----------------------
base_w = 15
base_h = 70

w_scale = (0.75, 1.5)
h_scale = (0.75, 1.25)

max_box = 50
min_box = 3
#----------------------


layout_path = 'datasets/wheat/custom_layout/'
layout_img_path = layout_path + 'img/'
os.makedirs(layout_img_path, exist_ok=True)
file = open(layout_path + 'bounding_boxes.txt', 'w')

for img_id in range(n_img):
    file.write(str(img_id) + '.png')
    
    n_box = np.random.randint(min_box, max_box)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    centers = np.random.randint(0, image_size, size=(n_box*100, 2))
    centers = centers[np.lexsort((centers[:, 1], centers[:, 0]))[::-1]]   # Make the box distribution more even
    for box_id in range(n_box):
        w = int(np.random.uniform(w_scale[0], w_scale[1]) * base_w)
        h = int(np.random.uniform(h_scale[0], h_scale[1]) * base_h)

        center = centers[box_id*100]
        angle = np.random.randint(-180, 180)

        # Create rotated rectangle
        rect = ((int(center[0]), int(center[1])), (w, h), angle)

        # Get the four vertices of the rectangle
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)

        color = [0, 0, 0]
        color[np.random.randint(3)] = 255 # Randomly set the channel of box 

        cv2.drawContours(image, [rect], 0, tuple(color), -1)


        x_min, x_max = np.min(rect[:, 0]), np.max(rect[:, 0])
        y_min, y_max = np.min(rect[:, 1]), np.max(rect[:, 1])
        box = np.array([x_min, y_min, x_max, y_max])
        box = np.clip(box, 0, image_size-1)
        file.write(f" {box[0]},{box[1]},{box[2]},{box[3]},0")

    file.write("\n")

    cv2.imwrite(layout_img_path + f"{img_id}.png", image)
