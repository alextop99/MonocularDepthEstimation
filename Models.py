import tensorflow as tf
from ModelBlocks import *

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
    
    def call(self, x):
        #* inputs - (6, x, y, 3)
        down1, pooled1 = self.downscale_blocks[0](x)
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

        #* return - (6, x, y, 1)
        return self.conv_layer(up4)
    
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