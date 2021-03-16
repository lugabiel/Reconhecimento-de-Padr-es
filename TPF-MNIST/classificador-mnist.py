import cv2
import csv

trainData = ''
line = ''

with open('trainReduzido.csv', newline ='\n') as trainCsv:
    trainData = csv.reader(trainCsv,delimiter=' ')
    for line in trainData:
        print(line)
    


