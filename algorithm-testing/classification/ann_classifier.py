import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np


def ann_trainer(x_train, y_train, x_val, y_val, params):
    epochs, batch_size = params

    # Model Template
    model = Sequential()  # declare model
    model.add(Dense(10, input_shape=(len(x_train[0]),), kernel_initializer='he_normal'))  # firstlayer
    model.add(Activation('tanh'))
    model.add(Dense(2048, activation=tf.nn.tanh))

    model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)

    loss = history.history['loss'][-1]

    return loss, model


def ann_classifier(x_test, model):
    return model.predict(x_test)