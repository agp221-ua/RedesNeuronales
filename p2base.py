import sys
from math import exp

class Neurona:
    def __init__(self, arraypesos, umbral):
        self.weights = arraypesos
        self.threshold = umbral

    def evaluate(self, entries):
        total = 0
        for i in range(0,len(entries)):
            total += entries[i] * self.weights[i]
        total += self.threshold
        return Neurona.act(total)

    @staticmethod
    def act(z):
        return 1/(1+exp(-z))

def sacarArraysForward():
    '''
    Funcion auxiliar que devuelve un array con todas las entradas y salidas
    :return: tupla con entradas y salidas
    '''
    ent = []
    sol = []

    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    ent.append((a, b, c, d))
                    sol.append(forward((a, b, c, d)))

    return ent, sol


def forward(t):
    '''
    Funcion forward que se pide el cual dada una entrada como tupla, devuelve la salida de la red
    :param t: tupla con la entrada
    '''

    N_And1 = Neurona([ 100,  100,    0, -100], -150)
    N_And2 = Neurona([-100, -100, -100,  100],  -50)
    N_And3 = Neurona([-100,  100, -100,  100], -150)
    N_And4 = Neurona([ 100, -100,  100, -100], -150)
    N_Or   = Neurona([ 100,  100,  100,  100],  -50)

    N_And1_out = N_And1.evaluate([t[0], t[1], t[2], t[3]])
    N_And2_out = N_And2.evaluate([t[0], t[1], t[2], t[3]])
    N_And3_out = N_And3.evaluate([t[0], t[1], t[2], t[3]])
    N_And4_out = N_And4.evaluate([t[0], t[1], t[2], t[3]])

    return N_Or.evaluate([N_And1_out, N_And2_out, N_And3_out, N_And4_out])

def kerasCalculate():
    '''
    Funcion que compila y prueba la red con keras
    '''
    from tensorflow import keras
    from keras import layers

    X, Y = sacarArraysForward()
    model = keras.Sequential()
    model.add(layers.Dense(4, input_dim=4, activation="sigmoid"))
    model.add(layers.Dense(1, input_dim=4, activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer='adam')

    model.fit(X, Y, epochs=4000, batch_size=16, verbose=0)

    k_sal = model.predict(x=X)

    for i in range(len(k_sal)):
        print(f'{X[i]} - {k_sal[i][0]}           {Y[i]}  -  {1 if k_sal[i][0] >= 0.5 else 0}')
    print('\n\n')

    ##### DESCOMENTAR PARA VER LOS PESOS
    # for capa in model.layers:
    #     w, b = capa.get_weights()
    #     for n in range(len(b)):
    #         print(f'w{n} = {w[n]}')
    #         print(f'umbral = {b[n]}')
    #     print()

if __name__ == '__main__':
    print(forward((1,1,1,1)))
    kerasCalculate()