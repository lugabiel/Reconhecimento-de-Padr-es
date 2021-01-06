# importing libraries

import math as mt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# funcao estimativa da densidade de 2 variaveis

def pdf2var(x,y,u1,u2,s1,s2,p):
    # i ,j : indices
    # u1,u2: media dataset 
    # s1,s2: desvio dataset
    return (1/(2*mt.pi*s1*s2*np.sqrt(1-(p**2))))*mt.exp((-(1)/(2*(1-(p**2)))))*((((x-u1)**2)/((s1**2))+(((y-u2)**2)/((s2)**2))-((2*p*(x-u1)*(y-u2))/(s1*s2))))



# defining media(mean)
mu = 0.5
# standard desvio padrao (standard deviation)
sigma = 0.1

# setup random module
np.random.seed(0)

numPoints = 200

# generating data set A
xgenA = np.random.normal(mu*8,sigma*6,(numPoints,1))
ygenA = np.random.normal(mu*8,sigma*6,(numPoints,1))

# calculando media (mgenA) e desvio (dgenA) estimado do data set A
mgenA = [np.mean(xgenA),np.mean(ygenA)]
dgenA = [np.std(xgenA) ,np.std(ygenA) ]
print(mgenA,dgenA)

# gen data set B
xgenB = np.random.normal(mu*4,sigma*6,(numPoints,1))
ygenB = np.random.normal(mu*4,sigma*6,(numPoints,1))

# calculando media e desvio estimado do data set B
mgenB = [np.mean(xgenB),np.mean(ygenB)]
dgenB = [np.std(xgenB) ,np.std(ygenB) ]
print(mgenB,dgenB)

# generate evenly spaced points
seqi = np.arange(0,6,0.12)
seqj = np.arange(0,6,0.12)

M1 = [np.zeros((len(seqi),len(seqj)))]
M2 = [np.zeros((len(seqi),len(seqj)))]
ci = 0
for i in seqi:
    ci = ci + 1
    cj = 0
    for j in seqj:
        cj = cj + 1
        M1[ci][cj] = pdf2var(i,j,mgenA[0],mgenA[1],dgenA[0],dgenA[1],0)

plt.scatter(xgenA,ygenA, color = 'r',marker =".")
plt.plot()
plt.scatter(xgenB,ygenB, color = 'g',marker ='.')
plt.show()