import tensorflow as tf
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from ModelBlocks import *
from DataGenerator import DataGeneratorAugmented
from dataset.kitti_dataset import KittiDatasetAugmented

tf.random.set_seed(123)

WIDTH = 256
HEIGHT = 64

LR = 0.001
EPOCHS = 10
BATCH_SIZE = 6

def main():

    kittiDataset = KittiDatasetAugmented("../dataset/kitti/")
    trainData, valData = kittiDataset.load_train_val()

    train_loader = DataGeneratorAugmented(
        data=trainData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    validation_loader = DataGeneratorAugmented(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    batch6X, batch6Y = train_loader.__getitem__(0)
    print(batch6X.shape)
    
    fig, ax = plt.subplots(6, 2, figsize=(50, 50))
    ax[0, 0].imshow(batch6X[0][0])
    ax[1, 0].imshow(batch6X[0][1])
    ax[2, 0].imshow(batch6X[0][2])
    ax[3, 0].imshow(batch6X[0][3])
    ax[4, 0].imshow(batch6X[0][4])
    ax[5, 0].imshow(batch6X[0][5])
    ax[0, 1].imshow(batch6X[1][0])
    ax[1, 1].imshow(batch6X[1][1])
    ax[2, 1].imshow(batch6X[1][2])
    ax[3, 1].imshow(batch6X[1][3])
    ax[4, 1].imshow(batch6X[1][4])
    ax[5, 1].imshow(batch6X[1][5])
    plt.show()
    

if __name__ == "__main__":
    main()