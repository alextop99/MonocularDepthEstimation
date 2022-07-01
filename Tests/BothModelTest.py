import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import matplotlib.gridspec as gridspec

import sys
sys.path.append('../')
from Loss import depth_loss_function

model = tf.keras.models.load_model('../model/model.tf', custom_objects = {"depth_loss_function": depth_loss_function})
modelAug = tf.keras.models.load_model('../model/modelAug.tf', custom_objects = {"depth_loss_function": depth_loss_function})

#image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png")
#image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000064.png")
#image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0023_sync/image_02/data/0000000018.png")
#image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_29/2011_09_29_drive_0026_sync/image_02/data/0000000005.png")
image_ = cv2.imread("../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000065.png")
image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
image_ = cv2.resize(image_, (256, 64))
image_ = tf.image.convert_image_dtype(image_, tf.float32)

#semantic_segmentation = cv2.imread("../dataset/kitti/semantic_segmentation/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png")
#semantic_segmentation = cv2.imread("../dataset/kitti/semantic_segmentation/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000064.png")
#semantic_segmentation = cv2.imread("../dataset/kitti/semantic_segmentation/2011_09_26/2011_09_26_drive_0023_sync/image_02/data/0000000018.png")
#semantic_segmentation = cv2.imread("../dataset/kitti/semantic_segmentation/2011_09_29/2011_09_29_drive_0026_sync/image_02/data/0000000005.png")
semantic_segmentation = cv2.imread("../dataset/kitti/semantic_segmentation/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000065.png")
semantic_segmentation = cv2.cvtColor(semantic_segmentation, cv2.COLOR_BGR2RGB)
semantic_segmentation = cv2.resize(semantic_segmentation, (256, 64))
semantic_segmentation = tf.image.convert_image_dtype(semantic_segmentation, tf.float32)

#depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005_depth.png", cv2.IMREAD_GRAYSCALE)
#depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000064_depth.png", cv2.IMREAD_GRAYSCALE)
#depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/val/2011_09_26_drive_0023_sync/proj_depth/groundtruth/image_02/0000000018_depth.png", cv2.IMREAD_GRAYSCALE)
#depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/val/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000005_depth.png", cv2.IMREAD_GRAYSCALE)
depth_map = cv2.imread("../dataset/kitti/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000065_depth.png", cv2.IMREAD_GRAYSCALE)

depth_map = cv2.resize(depth_map, (256, 64))
depth_map = np.expand_dims(depth_map, axis=2)
depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

x = np.empty((1, *(64, 256), 3))
x[0] = image_
output = model.predict(x)

start_time = time.time()
output = model.predict(x)
print("--- %s seconds ---" % (time.time() - start_time))

xAug = np.empty((2, 10, *(64, 256), 3))
xAug[0][0] = image_
xAug[1][0] = semantic_segmentation
outputAug = modelAug.predict(xAug)

start_time = time.time()
outputAug = modelAug.predict(xAug)
print("--- %s seconds ---" % (time.time() - start_time))

# fig, ax = plt.subplots(4, 2, figsize=(100, 100))
# ax[0, 0].title.set_text('Input Image')
# ax[0, 0].imshow(image_)
# ax[2, 0].title.set_text('GT Depth Map')
# ax[2, 0].imshow(depth_map)
# ax[3, 0].title.set_text('Original Model Depth Map')
# ax[3, 0].imshow(output[0])
# ax[0, 1].title.set_text('Input Image')
# ax[0, 1].imshow(image_)
# ax[1, 1].title.set_text('Semantic Segmented Image')
# ax[1, 1].imshow(semantic_segmentation)
# ax[2, 1].title.set_text('GT Depth Map')
# ax[2, 1].imshow(depth_map)
# ax[3, 1].title.set_text('Augmented Model Depth Map')
# ax[3, 1].imshow(outputAug[0])

# plt.show()

fig1 = plt.figure(constrained_layout=True)
gs = fig1.add_gridspec(3, 6)
f3_ax1 = fig1.add_subplot(gs[0, :3])
f3_ax1.set_title('Input Image')
f3_ax1.imshow(image_)
f3_ax1.set_axis_off()
f3_ax2 = fig1.add_subplot(gs[0, 3:])
f3_ax2.set_title('Semantic Segmentation')
f3_ax2.imshow(semantic_segmentation)
f3_ax2.set_axis_off()

f3_ax3 = fig1.add_subplot(gs[1, :])
f3_ax3.set_title('Ground Truth')
f3_ax3.imshow(depth_map)
f3_ax3.set_axis_off()

f3_ax4 = fig1.add_subplot(gs[2, :3])
f3_ax4.set_title('Base Output')
f3_ax4.imshow(output[0])
f3_ax4.set_axis_off()

f3_ax5 = fig1.add_subplot(gs[2, 3:])
f3_ax5.set_title('Augmented Output')
f3_ax5.imshow(outputAug[0])
f3_ax5.set_axis_off()

plt.show()


fig2 = plt.figure(constrained_layout=True)
gs = fig2.add_gridspec(2, 1)
f2_ax1 = fig2.add_subplot(gs[0, :])
f2_ax1.set_title('Base Neural Network')
f2_ax1.imshow(output[0])
f2_ax1.set_axis_off()
f2_ax2 = fig2.add_subplot(gs[1, :])
f2_ax2.set_title('Augmented Neural Network')
f2_ax2.imshow(outputAug[0])
f2_ax2.set_axis_off()

plt.show()