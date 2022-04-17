import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from Loss import depth_loss_function
from dataset.kitti_dataset import KittiDatasetAugmented
from DataGenerator import DataGeneratorAugmented

WIDTH = 256
HEIGHT = 64

BATCH_SIZE = 6

model = tf.keras.models.load_model('../model/modelAug.tf', custom_objects = {"depth_loss_function": depth_loss_function})
kittiDataset = KittiDatasetAugmented("../dataset/kitti/")
valData = kittiDataset.load_val()

validation_loader = DataGeneratorAugmented(
    data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
)

print(model.evaluate(validation_loader, batch_size=BATCH_SIZE))
