import cv2
import csv
import time as t

tInicial = t.time()

trainData = ''
line = ''

with open('trainReduzido.csv', newline ='\n') as trainCsv:
    trainData = csv.reader(trainCsv,delimiter=' ')
    lastline = -1
    for line in trainData:
        if (line[0][0] != lastline):
            lastline = line[0][0]
            print(line)
    


tFinal = t.time()
print(tFinal - tInicial)