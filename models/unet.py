import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self, img_height=256, img_width=448):
        super(UNet, self).__init__()
        self.img_height = img_height
        self.img_width = img_width

    def get_model(self):
        inputs = tf.keras.layers.Input(shape=[self.img_height, self.img_width, 6])
        e1 = self.encoder_block(inputs, 64)
        p1 = tf.keras.layers.MaxPooling2D(2)(e1)
        e2 = self.encoder_block(p1, 128)
        p2 = tf.keras.layers.MaxPooling2D(2)(e2)
        e3 = self.encoder_block(p2, 256)
        p3 = tf.keras.layers.MaxPooling2D(2)(e3)
        e4 = self.encoder_block(p3, 512)
        p4 = tf.keras.layers.MaxPooling2D(2)(e4)

        b = self.encoder_block(p4, 1024)
        b = self.channel_attention(b)

        skip_d1 = self.channel_attention(e4)
        skip_d2 = self.channel_attention(e3)
        skip_d3 = self.channel_attention(e2)
        skip_d4 = self.channel_attention(e1)

        d1 = self.decoder_block(b, skip_d1, 512)
        d2 = self.decoder_block(d1, skip_d2, 256)
        d3 = self.decoder_block(d2, skip_d3, 128)
        d4 = self.decoder_block(d3, skip_d4, 64)

        outputs = tf.keras.layers.Conv2D(3, 1, activation='sigmoid', padding='same')(d4)

        model = tf.keras.Model(inputs, outputs)
        return model
        
    def channel_attention(self, x, reduction=8):
        channels = x.shape[-1]

        gap = tf.keras.layers.GlobalAveragePooling2D()(x)

        fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')(gap)
        fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')(fc1)

        fc2 = tf.keras.layers.Reshape((1, 1, channels))(fc2)
        return x * fc2
    
    def encoder_block(self, x, filters, kernel_size=3, strides=1, padding='same'):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x
    
    def decoder_block(self, x, skip, filters, kernel_size=3, strides=1, padding='same'):
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding=padding)(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x