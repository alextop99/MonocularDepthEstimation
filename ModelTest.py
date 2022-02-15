import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt


#model = tf.keras.models.load_model('model/model.tf')
model = tf.keras.models.load_model('model/model2.tf')
print(model.summary())

image_ = cv2.imread("dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png")
image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
image_ = cv2.resize(image_, (256, 128))
image_ = tf.image.convert_image_dtype(image_, tf.float32)

depth_map = cv2.imread("dataset/kitti/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png", cv2.IMREAD_GRAYSCALE)
depth_map = cv2.resize(depth_map, (256, 128))
depth_map = np.expand_dims(depth_map, axis=2)
depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

x = np.empty((1, *(128, 256), 3))
x[0] = image_
output = model.predict(x)

print(depth_map)
print(output[0].shape)
print(output[0].dtype)

img_uint8 = output[0].astype(np.uint8)

fig, ax = plt.subplots(1, 4, figsize=(50, 50))
ax[0].imshow(image_)
ax[1].imshow(depth_map)
ax[2].imshow(output[0])
ax[3].imshow(img_uint8)

plt.show()