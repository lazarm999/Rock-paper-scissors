import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model, Sequential
from keras.callbacks import LearningRateScheduler
from keras.applications.mobilenet_v2 import MobileNetV2 

from utils import l_rate, dim, step_decay, data_generator

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

def train_model(train_dir, val_dir, model_name='model', epochs=25, batch_size=16, checkpoint=None):

    if checkpoint:
        model_path = os.path.join('models', f'{checkpoint}.h5')
        model = load_model(model_path)
    else:
        model_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(dim, dim, 3))
        model = Sequential()
        model.add(model_base)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=l_rate), metrics=['acc'])

    train_generator = data_generator(train_dir, batch_size=batch_size)
    val_generator = data_generator(val_dir, batch_size=batch_size, validation=True)
    
    lrate = LearningRateScheduler(step_decay)
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size,
                                  epochs=epochs,
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
    import argparse
    parser = argparse.ArgumentParser(description='This module is used for training the Rock, paper, scissors model.')
    parser.add_argument('train_data_path', type=str, help='The path to your training data directory, \
        which has 3 subdirectories correspoding to the rock, paper and scissors class.')
    parser.add_argument('val_data_path', type=str, help='The path to your validation data directory, \
    which has 3 subdirectories correspoding to the rock, paper and scissors class.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='The number of epochs for training the model, defaults to 25.')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size used for training, defaults to 16.')
    parser.add_argument('-m', '--model', type=str, default='model', help='Name of the model used for saving the model in h5 format in the models directory\
        after training is finished, defaults to \'model\'.')
    parser.add_argument('--cp', type=str, default=None, help='Name of the MobileNetv2 model already saved in h5 format in the models directory\
        from which to continue training, defaults to None, which means a randomly initialized model will be instantiated.')
    
    args = parser.parse_args()
    model, history = train_model(args.train_data_path, args.val_data_path, model_name=args.model, epochs=args.epochs,
        batch_size=args.batch_size, checkpoint=args.cp)
    plot_model_history(history)
    