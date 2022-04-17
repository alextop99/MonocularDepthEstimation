from dataset.kitti_dataset import KittiDatasetAugmented
import tensorflow as tf
from ModelBlocks import *
from Loss import depth_loss_function
from DataGenerator import DataGeneratorAugmented

tf.random.set_seed(123)

WIDTH = 256
HEIGHT = 64
# 1024 x 256

LR = 0.001
EPOCHS = 10
BATCH_SIZE = 6

class DepthEstimationAugmentedModel(tf.keras.Model):
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
        self.augmented_block = AugmentedBlock(filters[0])
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")

    def call(self, x):
        #* inputs - (2, 6, x, y, 3)
        down1, pooled1 = self.downscale_blocks[0](x[0])
        #* down1 - (6, x, y, 16)
        #* pooled1 - (6, x/2, y/2, 16)
        down2, pooled2 = self.downscale_blocks[1](pooled1)
        #* down2 - (6, x/2, y/2, 32)
        #* pooled2 - (6, x/4, y/4, 32)
        down3, pooled3 = self.downscale_blocks[2](pooled2)
        #* down3 - (6, x/4, y/4, 64)
        #* pooled3 - (6, x/8, y/8, 64)
        down4, pooled4 = self.downscale_blocks[3](pooled3)
        #* down4 - (6, x/8, y/8, 128)
        #* pooled4 - (6, x/16, y/16, 128)
        
        bn = self.bottle_neck_block(pooled4)
        #* (6, x/16, y/16, 256)

        up1 = self.upscale_blocks[0](bn, down4)
        #* up1 - (6, x/8, y/8, 128)
        up2 = self.upscale_blocks[1](up1, down3)
        #* up2 - (6, x/4, y/4, 64)
        up3 = self.upscale_blocks[2](up2, down2)
        #* up3 -  (6, x/2, y/2, 32)
        up4 = self.upscale_blocks[3](up3, down1)
        #* up4 -  (6, x, y, 16)

        #* aug - (6, x, y, 16)
        aug = self.augmented_block(up4, x[1])

        #* return - (6, x, y, 1)
        return self.conv_layer(aug)

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
    
    model = DepthEstimationAugmentedModel()
    
    # Compile the model
    model.compile(optimizer, loss=depth_loss_function)

    kittiDataset = KittiDatasetAugmented("dataset/kitti/")
    trainData, valData = kittiDataset.load_train_val()

    train_loader = DataGeneratorAugmented(
        data=trainData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    validation_loader = DataGeneratorAugmented(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    checkpoint_path = "checkpointsAug/cpAug.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
    
    
    model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=validation_loader,
        callbacks=[cp_callback]
    )
    
    model.save_weights("model/weightsAug.h5")
    model.save("model/modelAug.tf", save_format='tf')

if __name__ == "__main__":
    main()