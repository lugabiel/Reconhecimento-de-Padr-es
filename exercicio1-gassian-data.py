# <><> Gabriel Aragão - 2021 <><>
# TODO: a class for a dataset(std deviation,mean)
#     : 

# importing libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class dataset(object):
    '''dataset class for data generation'''
    def generate(self):
        self.x      = np.random.normal(self.desvio,self.media,(self.size,1))
        self.y      = np.random.normal(self.desvio,self.media,(self.size,1))
        pass
    
    def __init__(self,desvio=None,media=None,size=None):
        self.desvio = desvio
        self.media  = media
        self.size   = size
        self.x      = None
        self.y      = None
    def info(self):
        print(self.desvio)
        print(self.media)
        print(self.size)
        if input('print y/n?') == 'y':
            print(self.x,self.y)

        

# funcao densidade de probabilidade para 2 variaveis
def pdf2var(x,y,u1,u2,s1,s2,p):
    # x ,y : indices do ponto no espaco R^2
    # u1,u2: media media de cada variavel no dataset 
    # s1,s2: desvio padrao de cada variavel no dataset
    # p    : coeficiente de correlacao 
    A = (1/(2*np.pi*s1*s2*np.sqrt(1-(p**2))))
    B = ((-(1)/(2*(1-(p**2)))))
    C = ((x-u1)**2)/((s1**2))
    D = (((y-u2)**2)/((s2)**2))
    E = ((-2*p*(x-u1)*(y-u2))/(s1*s2)) #termo de covariancia
    return A*np.exp(B*(C+D+E))
    #return (1/(2*mt.pi*s1*s2*np.sqrt(1-(p**2))))*mt.exp(((-(1)/(2*(1-(p**2)))))*((((x-u1)**2)/((s1**2))+(((y-u2)**2)/((s2)**2))-((2*p*(x-u1)*(y-u2))/(s1*s2)))))

# mesuring probability surface for both datasets
def genSurface(mgenA,dgenA,mgenB,dgenB,correlacao,M1,M2):
    ci = 0
    mgenA[0] -= 0.03
    for i in seqi:
        ci = ci + 1
        cj = 0
        
        if mgenA[0] < 0:
           mgenA[0] = 7 
        for j in seqj:
            cj = cj + 1
            
            aux  = pdf2var(i,j,mgenA[0],mgenA[1],dgenA[0],dgenA[1],correlacao)
            aux2 = pdf2var(i,j,mgenB[0],mgenB[1],dgenB[0],dgenB[1],correlacao)
            #print(ci,cj,'--', aux, aux2)
            M1[0][ci-1][cj-1] = aux
            M2[0][ci-1][cj-1] = aux2
    return M1,M2

# defining mean and standard deviation
mu = 0.5
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
X,Y = np.meshgrid(seqi,seqj)

M1 = [np.zeros([len(seqi),len(seqj)])]
M2 = [np.zeros([len(seqi),len(seqj)])]

plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')

# Customize the z axis.
ax.set_zlim(0,1.5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# dataset params
correlacao = 0.2
steps = 20

# printing first frame of data sequence
M1, M2 = genSurface(mgenA,dgenA,mgenB,dgenB,correlacao, M1, M2)
#surface  = ax.plot_surface(X,Y,M1[0]+M2[0],cmap=cm.coolwarm,linewidth=0, antialiased=True)
plt.figure()
plt.scatter(xgenA,ygenA, color = 'r',marker =".")
plt.plot()
plt.scatter(xgenB,ygenB, color = 'g',marker ='.')

# axis for interpretation only
eixos = ['Dias até diagnóstico','Complexidade']
plt.title(eixos[1]+' vs '+eixos[1])
plt.xlabel(eixos[1])
plt.ylabel(eixos[0])

plt.show()
plt.pause(10)
    
# Add a color bar which maps values to colors.
fig.colorbar(surface, shrink=0.5, aspect=5)

while True:
    print('initialiazing',correlacao)
    for step in range(steps) :
        if correlacao > 0.9:
            correlacao = 0.1
        print('redrwaing',correlacao)
        # mesuring probability surface for both datasets 
        M1, M2 = genSurface(mgenA,dgenA,mgenB,dgenB,correlacao, M1, M2)
        
        #  gen dataset A & B
        #plt.scatter(xgenA,ygenA, color = 'r',marker =".")
        #plt.scatter(xgenB,ygenB, color = 'g',marker ='.')
    
        plt.pause(0.01)
        # printing gen dataset 3d
        ax.clear()
        ax.set_zlim(0,1.0)
        ax.set_title('correlacao = '+str(correlacao))
        surface  = ax.plot_surface(X,Y,M1[0]+M2[0],cmap=cm.coolwarm,linewidth=0, antialiased=True)
        correlacao += 0.025
    plt.show()
