import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model, Sequential
from keras.callbacks import LearningRateScheduler
from keras.applications.mobilenet_v2 import MobileNetV2 
from keras.regularizers import l2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

l_rate = 0.0007
height = 300
width = 200

def step_decay(epoch):
    drop = 0.5
    epochs_drop = 10.0
    lrate = l_rate * np.math.pow(drop, np.math.floor((1+epoch)/epochs_drop))
    return lrate

def data_generator(data_path, batch_size=16, validation=False):
    if validation:
        datagen = ImageDataGenerator(
            rescale=1./255)        
    else:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=(0.6, 1.6),
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')
    gen = datagen.flow_from_directory(
        data_path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical')
    return gen

def train_model(train_dir, val_dir, model_name='model', epochs=25, batch_size=16, checkpoint=None):

    if checkpoint:
        model_path = os.path.join('models', f'{checkpoint}.h5')
        model = load_model(model_path)
    else:
        model_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(height, width, 3))
        model = Sequential()
        model.add(model_base)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=l_rate), metrics=['acc'])

    train_generator = data_generator(train_dir)
    val_generator = data_generator(val_dir, validation=True)
    
    lrate = LearningRateScheduler(step_decay)
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size,
                                  epochs=25,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.n // batch_size,
                                  verbose=2,
                                  callbacks=[lrate],
                                  shuffle=True)

    model.save_weights(os.path.join('models', f'{model_name}_weights.h5'))
    model.save(os.path.join('models', f'{model_name}.h5'))
    return model, history

def plot_model_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    # Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_dir = r'C:\Users\vukbibic\Desktop\Papir kamen makaze veci\data\dataset'
    val_dir = r'C:\Users\vukbibic\Desktop\Papir kamen makaze veci\data\test images'
    model, history = train_model(train_dir, val_dir, batch_size=8)
    plot_model_history(history)
    