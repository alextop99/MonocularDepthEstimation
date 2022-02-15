from dataset.kitti_dataset import KittiDataset
import tensorflow as tf
from ModelBlocks import *

import numpy as np
import cv2
import datetime
from matplotlib import pyplot as plt

tf.random.set_seed(123)

HEIGHT = 128
WIDTH = 256
LR = 0.0002
EPOCHS = 10
BATCH_SIZE = 6

#* Data Generator
#* Reads data from the images loaded by the KittiDataset library and preprocesses them
class DataGenerator(tf.keras.utils.Sequence):
    #* Initialize the data generator with specific parameters
    def __init__(self, data, batch_size=6, dim=(370, 1240), n_channels=3, shuffle=True):
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
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
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
        image_ = cv2.resize(image_, self.dim[::-1])
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_map = cv2.resize(depth_map, self.dim[::-1])
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    #* Get image paths and load them
    def data_generation(self, batch):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth_map"][batch_id],
            )

        return x, y

class DepthEstimationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
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
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="tanh")

    def calculate_loss(self, target, pred):
        # Edges
        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
            abs(smoothness_y)
        )

        # Structural similarity (SSIM) index
        ssim_loss = tf.reduce_mean(
            1
            - tf.image.ssim(
                target, pred, max_val=WIDTH, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
            )
        )
        # Point-wise depth
        l1_loss = tf.reduce_mean(tf.abs(target - pred))

        loss = (
            (self.ssim_loss_weight * ssim_loss)
            + (self.l1_loss_weight * l1_loss)
            + (self.edge_loss_weight * depth_smoothness_loss)
        )

        return loss

    @property
    def metrics(self):
        return [self.loss_metric] + self.compiled_metrics.metrics 

    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.calculate_loss(target, pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_metric.update_state(loss)
        self.compiled_metrics.update_state(target, pred)
        
        res = {}
        custom_metrics = {
            "loss": self.loss_metric.result(),
        }
        compile_metrics = {m.name: m.result() for m in self.metrics}
        res.update(custom_metrics)
        res.update(compile_metrics)
        return res

    def test_step(self, batch_data):
        input, target = batch_data

        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        self.compiled_metrics.update_state(target, pred)
        
        res = {}
        custom_metrics = {
            "loss": self.loss_metric.result(),
        }
        compile_metrics = {m.name: m.result() for m in self.metrics}
        res.update(custom_metrics)
        res.update(compile_metrics)
        return res

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
    
    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    # Compile the model
    model.compile(optimizer, loss=cross_entropy, metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])

    kittiDataset = KittiDataset("dataset/kitti/")
    trainData, valData = kittiDataset.load_train_val()

    train_loader = DataGenerator(
        data=trainData.reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
    )
    validation_loader = DataGenerator(
        data=valData.reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
    )
    
    checkpoint_path = "checkpoints/cp.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=validation_loader,
        callbacks=[cp_callback, tensorboard_callback]
    )
    
    model.save_weights("model/weights.h5")
    model.save("model/model.tf", save_format='tf')
    print(model.summary())
    
    plt.plot(history.history['mse'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['root_mean_squared_error'])
    plt.show()

if __name__ == "__main__":
    main()