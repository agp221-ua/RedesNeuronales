from math import exp


class Neurona:
    def __init__(self, w, t):
        self.weights = w
        self.threshold = t

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
        aux = input('Dame una a y b: ')
        aux = [int(x) for x in aux.split(' ')]
        print(f'El res = {Neu.use([aux[0], aux[1]])}')

def main():
    Neu_or = Neurona([58,58], -28)
    Neu_and = Neurona([58,58], -86)
    Neu_nand = Neurona([-100,-100], 30)
    #probaramano()






if __name__ == '__main__':
    main()