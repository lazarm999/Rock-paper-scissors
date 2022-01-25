import numpy as np
from keras.preprocessing.image import ImageDataGenerator

l_rate = 0.0007
dim = 192

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
        target_size=(dim, dim),
        batch_size=batch_size,
        class_mode='categorical')
    return gen

def get_prediction_text(prediction):
    maks = max(prediction[0][0], prediction[0][1], prediction[0][2])
    if maks == prediction[0][0]:
        return "papir"
    elif maks == prediction[0][1]:
        return "kamen"
    else:
        return "makaze"