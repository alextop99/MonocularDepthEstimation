from tensorflow.keras import layers

class DownscaleBlock(layers.Layer):
    #* Stride 1 and padding = "same" => output_shape = input_shape
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2A = layers.BatchNormalization()
        self.bn2B = layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        #* input_tensor - (6, x, y, z)
        d = self.convA(input_tensor)
        #* d - (6, x, y, filters)
        x = self.bn2A(d)
        #* x - (6, x, y, filters)
        x = self.reluA(x)
        #* x - (6, x, y, filters)

        x = self.convB(x)
        #* x - (6, x, y, filters)
        x = self.bn2B(x)
        #* x - (6, x, y, filters)
        x = self.reluB(x)
        #* x - (6, x, y, filters)

        x += d
        #* x - (6, x, y, filters)
        p = self.pool(x)
        #* p - (6, x/2, y/2, filters)
        return x, p


class UpscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2A = layers.BatchNormalization()
        self.bn2B = layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        #* x - (6, x, y, z)
        x = self.us(x)
        #* x - (6, x * 2, y * 2, z)
        #* skip - (6, x * 2, y * 2, w)
        concat = self.conc([x, skip])
        #* x - (6, x * 2, y * 2, z + w)
        x = self.convA(concat)
        #* x - (6, x * 2, y * 2, filters)
        x = self.bn2A(x)
        #* x - (6, x * 2, y * 2, filters)
        x = self.reluA(x)
        #* x - (6, x * 2, y * 2, filters)

        x = self.convB(x)
        #* x - (6, x * 2, y * 2, filters)
        x = self.bn2B(x)
        #* x - (6, x * 2, y * 2, filters)
        x = self.reluB(x)
        #* x - (6, x * 2, y * 2, filters)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        #* x - (6, x, y, z)
        x = self.convA(x)
        #* x - (6, x, y, filters)
        x = self.reluA(x)
        #* x - (6, x, y, filters)
        x = self.convB(x)
        #* x - (6, x, y, filters)
        x = self.reluB(x)
        #* x - (6, x, y, filters)
        
        return x
