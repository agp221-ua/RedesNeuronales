import sys
from math import exp
from tensorflow import keras
from keras import layers

class Neurona:
    def __init__(self, arraypesos, umbral):
        self.weights = arraypesos
        self.threshold = umbral

    def use(self, entries):
        total = 0
        for i in range(0,len(entries)):
            total += entries[i] * self.weights[i]
        total += self.threshold
        return self.act(total)
        #return 1 if 0.9999999999999 < self.act(total) <= 1 else 0 if 0 <= self.act(total) < 0.000000000001 else -1

    def act(self, z):
        return 1/(1+exp(-z))

def probaramano(Neu):
    while True:
        aux = input('Dame una a, b, c y d: ')
        aux = [int(x) for x in aux.split(' ')]
        print(f'El res = {Neu.use(aux)}')

def sacarSolucionesCon4(Neu):
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    print(f'- {a}{b}{c}{d} - {Neu.use([a,b,c,d])}')


def sacarSolucionesFordward():
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    print(f'- {a}{b}{c}{d} - {fordward((a,b,c,d))}')
def sacarSolucionesFordwardK(ksal):
    print('ENTRADA   ESPERADA   OBTENIDA')
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    print(f'- {a}{b}{c}{d}   -   {round(fordward((a,b,c,d)))}   -   {ksal[8*a + 4*b + 2*c + d][0]}  \t{"OK" if (ksal[8*a + 4*b + 2*c + d][0] > 0.5 and round(fordward((a,b,c,d))) == 1) or (ksal[8*a + 4*b + 2*c + d][0] < 0.5 and round(fordward((a,b,c,d))) == 0) else "BAD"}')


def sacarArraysFordward():
    ent = []
    sol = []

    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    ent.append((a,b,c,d))
                    sol.append(fordward((a,b,c,d)))

    return ent, sol


def fordward(t):

    N_And1 = Neurona([50, 50, 0, -300], -75)
    N_And2 = Neurona([-100, -100, -100, 100], -50)
    N_And3 = Neurona([-100, 100, -100, 100], -150)
    N_And4 = Neurona([100, -100, 100, -100], -150)
    N_Or = Neurona([500, 500, 500, 500, ], -250)

    return N_Or.use([N_And1.use([t[0],t[1],t[2],t[3]]),
             N_And2.use([t[0],t[1],t[2],t[3]]),
             N_And3.use([t[0],t[1],t[2],t[3]]),
             N_And4.use([t[0],t[1],t[2],t[3]])])
def main():
    if len(sys.argv) != 2:
        print('ERROR: the arguments is not correct')
        return
    #Neu_or = Neurona([500,500,500,500,], -250) #Ta bueno
    #Neu_and = Neurona([100,100,100,100,], -350) #Ta bueno
    #Neu_nand = Neurona([-500,-500,-500,-500,], 1750) #Ta bueno
    # model = keras.Sequential([
    #     keras.Input(shape=(4,)),
    #     layers.Dense(1,activation="sigmoid"),
    #     layers.Dense(1,activation="sigmoid")
    # ])
    # model.compile(loss="mean_squared_error", optimizer="adam")
    ent, sal = sacarArraysFordward()
    # model.fit(
    #     x=ent,
    #     y=sal,
    #     batch_size=8,
    #     epochs=20
    # )
    model = keras.Sequential()
    model.add(layers.Dense(4, input_dim=4, activation="sigmoid"))
    model.add(layers.Dense(1, input_dim=4, activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(ent, sal, epochs=int(sys.argv[1]), batch_size=16, verbose=0)
    k_sal = model.predict(x=ent)
    sacarSolucionesFordwardK(k_sal)

if __name__ == '__main__':
    main()