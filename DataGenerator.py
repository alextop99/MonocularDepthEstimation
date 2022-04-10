import tensorflow as tf
import cv2
import numpy as np

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
    
#* Data Generator
#* Reads data from the images loaded by the KittiDataset library and preprocesses them
class DataGeneratorAugmented(DataGenerator):
    #* Initialize the data generator with specific parameters
    def __init__(self, data, batch_size=6, dim=(1240, 370), n_channels=3, shuffle=True):
        super().__init__(data, batch_size, dim, n_channels, shuffle)

    #* Load images from paths
    def load(self, image_path, semantic_segmenation_path, depth_path):
        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)
        
        semantic_segmentation = cv2.imread(semantic_segmenation_path)
        semantic_segmentation = cv2.cvtColor(semantic_segmentation, cv2.COLOR_BGR2RGB)
        semantic_segmentation = cv2.resize(semantic_segmentation, self.dim)
        semantic_segmentation = tf.image.convert_image_dtype(semantic_segmentation, tf.float32)

        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)
        
        return (image_, semantic_segmentation), depth_map

    #* Get image paths and load them
    def data_generation(self, batch):
        x = np.empty((self.batch_size, 2, *(self.dim[::-1]), self.n_channels))
        y = np.empty((self.batch_size, *(self.dim[::-1]), 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["semantic_segmentation"][batch_id],
                self.data["depth_map"][batch_id],
            )
            
        x = np.array_split(x, 2, axis=1)
        x[0] = np.squeeze(x[0], axis=1)
        x[1] = np.squeeze(x[1], axis=1)

        return np.array(x), y