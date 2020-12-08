import tensorflow as tf
import os

class image_reader:
    def __init__(self, path):
        self.path = path

    def train_generator(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                        horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(os.path.join(self.path, 'training'),
                                                            target_size=(224, 224),
                                                            batch_size=100,
                                                            class_mode='binary')
        return train_generator

    def val_generator(self):
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow_from_directory(os.path.join(self.path, 'validation'),
                                                        target_size=(224, 224),
                                                        batch_size=20,
                                                        class_mode='binary')
        return val_generator

