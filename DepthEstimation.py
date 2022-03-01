from dataset.kitti_dataset import KittiDataset
import tensorflow as tf
from ModelBlocks import *
from Loss import depth_loss_function

import numpy as np
import cv2


tf.random.set_seed(123)

WIDTH = 256
HEIGHT = 64

LR = 0.001
EPOCHS = 1
BATCH_SIZE = 6

#* Data Generator
#* Reads data from the images loaded by the KittiDataset library and preprocesses them
class DataGenerator(tf.keras.utils.Sequence):
    #* Initialize the data generator with specific parameters
    def __init__(self, data, batch_size=6, dim=(1240, 370), n_channels=3, shuffle=True):
        self.data= data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    #* Get the number of batches
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    #* Get a specific batch
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    #* Load images from paths
    def load(self, image_path, depth_path):
        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)
        
        return image_, depth_map

    #* Get image paths and load them
    def data_generation(self, batch):
        x = np.empty((self.batch_size, *(self.dim[::-1]), self.n_channels))
        y = np.empty((self.batch_size, *(self.dim[::-1]), 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth_map"][batch_id],
            )

        return x, y

class DepthEstimationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        filters = [16, 32, 64, 128, 256]
        self.downscale_blocks = [
            DownscaleBlock(filters[0]),
            DownscaleBlock(filters[1]),
            DownscaleBlock(filters[2]),
            DownscaleBlock(filters[3]),
        ]
        self.bottle_neck_block = BottleNeckBlock(filters[4])
        self.upscale_blocks = [
            UpscaleBlock(filters[3]),
            UpscaleBlock(filters[2]),
            UpscaleBlock(filters[1]),
            UpscaleBlock(filters[0]),
        ]
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")

    #TODO: Generate the semantic segmentation of the images and pass them to a layer (x is the batch of images)
    #? Should I concatenate the semantic segmenation to the input of the layer?
    #? Should the shapes be the same?
    def call(self, x):
        #* inputs - (6, 128, 256, 3)
        down1, pooled1 = self.downscale_blocks[0](x)
        #* down1 - (6, 128, 256, 16)
        #* pooled1 - (6, 64, 128, 16)
        down2, pooled2 = self.downscale_blocks[1](pooled1)
        #* down2 - (6, 64, 128, 32)
        #* pooled2 - (6, 32, 64, 32)
        down3, pooled3 = self.downscale_blocks[2](pooled2)
        #* down3 - (6, 32, 64, 64)
        #* pooled3 - (6, 16, 32, 64)
        down4, pooled4 = self.downscale_blocks[3](pooled3)
        #* down4 - (6, 16, 32, 128)
        #* pooled4 - (6, 8, 16, 128)
        
        bn = self.bottle_neck_block(pooled4)
        #* (6, 8, 16, 256)

        up1 = self.upscale_blocks[0](bn, down4)
        #* up1 - (6, 16, 32, 128)
        up2 = self.upscale_blocks[1](up1, down3)
        #* up2 - (6, 32, 64, 64)
        up3 = self.upscale_blocks[2](up2, down2)
        #* up3 -  (6, 64, 128, 32)
        up4 = self.upscale_blocks[3](up3, down1)
        #* up4 -  (6, 128, 256, 16)

        #* return - (6, 128, 256, 1)
        return self.conv_layer(up4)

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR,
        amsgrad=False,
    )
    
    model = DepthEstimationModel()
    
    # Compile the model
    model.compile(optimizer, loss=depth_loss_function)

    kittiDataset = KittiDataset("dataset/kitti/")
    trainData, valData = kittiDataset.load_train_val()

    train_loader = DataGenerator(
        data=trainData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    validation_loader = DataGenerator(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    checkpoint_path = "checkpoints/cp.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
    
    
    model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=validation_loader,
        callbacks=[cp_callback]
    )
    
    model.save_weights("model/weights.h5")
    model.save("model/model.tf", save_format='tf')

if __name__ == "__main__":
    main()