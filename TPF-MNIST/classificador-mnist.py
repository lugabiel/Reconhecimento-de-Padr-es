import cv2
import csv
import time as t
from PIL import Image as img

tInicial = t.time()

trainData = ''
line = ''

with open('trainReduzido.csv', newline ='\n') as trainCsv:
    trainData = csv.reader(trainCsv,delimiter=' ')
    lastline = -1
    for line in trainData:
        if (line[0][0] != lastline):
            lastline = line[0][0]
            print(line[0])



tFinal = t.time()
print(tFinal - tInicial)