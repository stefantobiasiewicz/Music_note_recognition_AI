# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from keras.testing_utils import layer_test
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# link do goscia od kogo to robiłem
# https://www.youtube.com/watch?v=6_2hzRopPbQ

if __name__ == '__main__':
    #   Przygotowanie danych do do terenowania i testowania
    df = pd.read_csv('../Notes/data1.csv')
    # df.head(1)
    X = pd.get_dummies(df.drop(['label'],
                               axis=1))  # ?(moja interpretacja)? stworzenie tablicy 2 wymiarowej zawierającej to co w csv z wyłączeniem kolumny label
    X = X / 256
    # X
    # y = df['label'] # stworzenie tablicy zawierajacej label ale w string, poniżej konwertuje każdą z nutek na cyferke
    mapping = {k: v for v, k in enumerate(df.label.unique())}
    y = df.label.map(mapping)
    y = y / 4
    # y  y[3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    #   Tworzenie modelu
    model = Sequential()

    model.add(Dense(32, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #   Szkolenie
    model.fit(X_train, y_train, epochs=2, batch_size=32)

    #   Predykcja
    y_hat = model.predict(X_test)

    #   Zapis i odczyt
    # model.save('model.h5', save_format='h5')
    # model = load_model('model.h5')


    num_classes = 5
    model = Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

