import tensorflow as tf
import Data_reader
import PIL
from tensorflow.keras.utils import Sequence

path = '/Users/nikhilshah/Desktop'
reader = Data_reader.image_reader(path)
train = reader.train_generator()
validation = reader.val_generator()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train,
                    steps_per_epoch=33,
                    epochs=5,
                    validation_data=validation,
                    validation_steps=40,
                    verbose=1)

model.save('saved_model/my_model')
