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
        #* input_tensor - (6, 128, 256, 3)
        d = self.convA(input_tensor)
        #* d - (6, 128, 256, 16)
        x = self.bn2A(d)
        #* x - (6, 128, 256, 16)
        x = self.reluA(x)
        #* x - (6, 128, 256, 16)

        x = self.convB(x)
        #* x - (6, 128, 256, 16)
        x = self.bn2B(x)
        #* x - (6, 128, 256, 16)
        x = self.reluB(x)
        #* x - (6, 128, 256, 16)

        x += d
        #* x - (6, 128, 256, 16)
        p = self.pool(x)
        #* p - (6, 64, 128, 16)
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
        #* x - (6, 8, 16, 256)
        x = self.us(x)
        #* x - (6, 16, 32, 256)
        #* skip - (6, 16, 32, 128)
        concat = self.conc([x, skip])
        #* x - (6, 16, 32, 384)
        x = self.convA(concat)
        #* x - (6, 16, 32, 128)
        x = self.bn2A(x)
        #* x - (6, 16, 32, 128)
        x = self.reluA(x)
        #* x - (6, 16, 32, 128)

        x = self.convB(x)
        #* x - (6, 16, 32, 128)
        x = self.bn2B(x)
        #* x - (6, 16, 32, 128)
        x = self.reluB(x)
        #* x - (6, 16, 32, 128)

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
        #* x - (6, 8, 16, 128)
        x = self.convA(x)
        #* x - (6, 8, 16, 256)
        x = self.reluA(x)
        #* x - (6, 8, 16, 256)
        x = self.convB(x)
        #* x - (6, 8, 16, 256)
        x = self.reluB(x)
        #* x - (6, 8, 16, 256)
        
        return x
