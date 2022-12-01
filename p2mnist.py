from tensorflow import keras
from keras import layers
import numpy as np

def main():
    # array de 60.000 arrays de 28x28 con un float de 0-255
    enttemporal = keras.datasets.mnist.load_data()
    print('[LOG] Datos cargados.')
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('[LOG] Datos convertidos')
    model = keras.Sequential()
    model.add(layers.Input(shape=784))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10,activation="softmax"))
    print('[LOG] Red creada')
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print('[LOG] Red compilada')
    sal=None
    input('Continuar: ')
    model.fit(x_train, y_train
              , batch_size=128, epochs=1, validation_split=0.1)
    print('[LOG] To calculao')



if "__main__" == __name__:
    main()
