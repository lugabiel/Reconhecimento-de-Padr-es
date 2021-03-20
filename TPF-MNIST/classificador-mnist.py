
import csv
import time as t

# MNIST dataset padrao da bib KERAS
from keras.datasets import mnist
(xTreino,yTreino) , (xTeste,yTeste) = mnist.load_data()



# tentando abrir imagens
import matplotlib.pyplot as plt
from PIL import Image as img

# inicia timer
tInicial = t.time()

# abrindo arq csv e printa somente a primeira linha da cada novo rotulo
trainData = ''
line = ''
with open('trainReduzido.csv', newline ='\n') as trainCsv:
    trainData = csv.reader(trainCsv,delimiter=' ')
    lastline = -1
    for line in trainData:
        if (line[0][0] != lastline):
            lastline = line[0][0]
            print(line[0], lastline  )

# abrindo imagem a partir do csv
img_index = 35
print(yTreino[img_index])
print(xTreino[img_index])
plt.imshow(xTreino[img_index],cmap='Greys')
plt.show()
# finaliza e printa tempo de execucao
tFinal = t.time()
print(tFinal - tInicial)