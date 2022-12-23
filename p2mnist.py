import sys
import time

from tensorflow import keras
from keras import layers
import numpy as np

FOTONUM = 60000
OPTION = False

def prepareData(arrayentrada, flaten, fotonum):
    aux = []
    for foto in arrayentrada:
        aux.append(np.array(foto).flatten().tolist() if flaten else np.array(foto).tolist())
    entt = []
    for i in range(fotonum):
        entt.append(aux[i])
    aux = np.array(entt)

    return aux

def translateResult(res):
    aux = []
    for foto in res:
        try:
            if OPTION:
                best = (-1, 0)
                cont = 0
                for label in foto:
                    if label > best[1]:
                        best = (cont, label)
                    cont += 1
                assert(best[0] != -1)
                aux.append(best[0])
            else:
                aux.append(np.where(foto > 0.5)[0][0])
        except:
            aux.append(-1)
    return aux

def mostrarComparacion(esperado, obtenido):
    acertados = 0
    total = len(esperado)
    print(f'\n\t\t##### RESULTADOS #######')
    print(f'\n\tESPERADO \tOBTENIDO \tACERTADO')
    print(f'\t----------------------------------')
    for i in range(len(esperado)):
        acertado = esperado[i] == obtenido[i]
        print(f'\t    {esperado[i]}\t\t   {obtenido[i] if obtenido[i] != -1 else "X"}\t\t   {"OK" if acertado else "FAIL" if obtenido[i] != -1 else "ERROR"}')
        acertados += 1 if acertado else 0
    print(f'\t----------------------------------')
    print(f'\t PORCENTAJE DE ACIERTO:  {round(acertados/total * 100, 2)}%')

def main():
    if not (2 <= len(sys.argv) <= 3):
        print('ERROR: the arguments is not correct')
        print('USE:  python p2mnist.py <num epochs> [first|best]')
        return
    if len(sys.argv) == 3:
        global OPTION
        OPTION = sys.argv[2] == 'best'
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
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    ent = np.array([a.flatten() for a in x_train])
    sal = prepareData(y_train, False, 60000)

    enttest = np.array([a.flatten() for a in x_test])
    saltest = prepareData(y_test, False, 10000)


    print('[LOG] Datos convertidos')
    model = keras.Sequential()
    model.add(layers.Input(shape=784))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10,activation="softmax"))
    print('[LOG] Red creada')
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print('[LOG] Red compilada')
    print('[LOG] Empezando calculo')
    t1 = time.time()
    epochs = 15
    model.fit(ent, sal, batch_size=128, epochs=epochs, validation_split=0.1)
    print(f'[LOG] To calculao en {time.time() - t1} seg con epoch = {epochs}')

    obtenido = translateResult(model.predict(x=enttest))
    esperado = translateResult(saltest)
    mostrarComparacion(esperado, obtenido)


if "__main__" == __name__:
    main()
