# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:59:56 2023

@author: MOODY
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# definir el modelo
model = Sequential()

# añadir la capa de convolución
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# añadir la capa de pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# aplanar las capas
model.add(Flatten())

# añadir una capa oculta completamente conectada
model.add(Dense(units=128, activation='relu'))

# añadir la capa de salida
model.add(Dense(units=10, activation='softmax'))

# compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
