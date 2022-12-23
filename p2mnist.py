import random
import sys
import time

from tensorflow import keras
from keras import layers
import numpy as np

FOTONUM = 60000
OPTION = False

def get1000RandImg(e, s):
    '''
    Funcion auxiliar para obtener 1000 imagenes aleatorias
    :param e: array del que obtener las imagenes
    :param s: array de donde conseguir las salidas esperadas
    :return: nuevos arrays con las 1000 imagenes
    '''
    posiciones_probadas = []
    ent_mil_rand = []
    sal_mil_rand = []
    while len(posiciones_probadas) < 1000:
        aux = random.randint(0, 59999)
        if aux in posiciones_probadas:
            continue
        ent_mil_rand.append(e[aux])
        sal_mil_rand.append(s[aux])
        posiciones_probadas.append(aux)
    ent_mil_rand = np.array(ent_mil_rand)
    sal_mil_rand = np.array(sal_mil_rand)
    return ent_mil_rand, sal_mil_rand


def mainMejorado():
    '''
    Main preparado para compilar y mostrar los resultados de la mejor red que se ha conseguido
    '''
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    ent = np.array([a.flatten() for a in x_train])
    sal = keras.utils.to_categorical(y_train, num_classes)

    enttest = np.array([a.flatten() for a in x_test])
    saltest = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential()
    model.add(layers.Input(shape=784))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.55), metrics=["accuracy"])

    model.fit(ent, sal, batch_size=128, epochs=20, validation_split=0.1)

    loss, accuracy = model.evaluate(x=enttest, y=saltest)

    print('Resultados Test')
    print(f'Valor de perdida: {loss}')
    print(f'Precision: {accuracy}')
    print(f'Error de clasificacion: {1 - accuracy}\n\n')



def main():
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    ent = np.array([a.flatten() for a in x_train])
    sal = keras.utils.to_categorical(y_train, num_classes)

    enttest = np.array([a.flatten() for a in x_test])
    saltest = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential()
    model.add(layers.Input(shape=784))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(ent, sal, batch_size=128, epochs=15, validation_split=0.1)

    loss, accuracy = model.evaluate(x=enttest, y=saltest)

    print('Resultados Test')
    print(f'Valor de perdida: {loss}')
    print(f'Precision: {accuracy}')
    print(f'Error de clasificacion: {1 - accuracy}\n\n')

    ent_mil_rand, sal_mil_rand = get1000RandImg(ent, sal)
    loss, accuracy = model.evaluate(x=ent_mil_rand, y=sal_mil_rand)

    print('Resultados 1000 rand ENTRENAMIENTO')
    print(f'Valor de perdida: {loss}')
    print(f'Precision: {accuracy}')
    print(f'Error de clasificacion: {1 - accuracy}\n\n')

    ent_mil_rand, sal_mil_rand = get1000RandImg(ent, sal)
    loss, accuracy = model.evaluate(x=ent_mil_rand, y=sal_mil_rand)

    print('Resultados 1000 rand TEST')
    print(f'Valor de perdida: {loss}')
    print(f'Precision: {accuracy}')
    print(f'Error de clasificacion: {1 - accuracy}\n\n')



if "__main__" == __name__:
    main()
    # mainMejorado()
