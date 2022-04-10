import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from Loss import depth_loss_function

model = tf.keras.models.load_model('../model/model.tf', custom_objects = {"depth_loss_function": depth_loss_function})

#image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png")
#image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000064.png")
image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0023_sync/image_02/data/0000000018.png")
image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
image_ = cv2.resize(image_, (256, 64))
image_ = tf.image.convert_image_dtype(image_, tf.float32)

#depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005_depth.png", cv2.IMREAD_GRAYSCALE)
#depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000064_depth.png", cv2.IMREAD_GRAYSCALE)
depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/val/2011_09_26_drive_0023_sync/proj_depth/groundtruth/image_02/0000000018_depth.png", cv2.IMREAD_GRAYSCALE)
depth_map = cv2.resize(depth_map, (256, 64))
depth_map = np.expand_dims(depth_map, axis=2)
depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

x = np.empty((1, *(64, 256), 3))
x[0] = image_
output = model.predict(x)

fig, ax = plt.subplots(1, 3, figsize=(50, 50))
ax[0].imshow(image_)
ax[1].imshow(depth_map)
ax[2].imshow(output[0])

plt.show()
