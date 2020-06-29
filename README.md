# Tarea-3
## Modelos Probabilsticos de Señales y Sistemas
### Esteban Leonardo Rodríguez Quintana
### B66076

## Solución:

""" Imports """
import numpy as np


import matplotlib.pyplot as plt  # para creación de gráfias variadas
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pandas import *  # usada para manejo del archivo lote en formato csv
from scipy.stats import norm
from scipy.optimize import curve_fit


lote = np.loadtxt('xy.csv',delimiter = ',')
print(lote)
### (25 %) A partir de los datos, encontrar la mejor curva de ajuste (modelo 
###  probabilístico) para las funciones de densidad marginales de X y Y.

#### Funciones marginales: axis = 0 (columnas)
marginalesY = np.sum(lote, axis=0)
print('\n   -> Marginales para y: \n')
print(marginalesY)

print('\n   -> Marginales para x \n')
marginalesX = np.sum(lote, axis=1)
print(marginalesX)

#### Plot de marginales Y:
ys = np.linspace(5,1,21)
plt.figure(0)
plt.plot(ys,marginalesY)
plt.savefig('img/maginales y')


#### Con curva de ajuste Normal:
n = len(marginalesY) 
meanY = sum(ys*marginalesY)/n                   #note this correction
sigmaY = sum(marginalesY*(ys-meanY)**2)/n 

def gaus(x,a,x0,sigma):
       return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,ys,marginalesY,p0=[1,meanY,sigmaY])
plt.figure(1)
plt.plot(ys,marginalesY,'b:',label='data')
plt.plot(ys,gaus(ys,*popt),'r',label='fit')
plt.legend()
plt.savefig('img/maginales y con ajuste')

#### Plot de marginales X:
xs = np.linspace(5,1,11)
plt.figure(2)
plt.plot(xs,marginalesX)
plt.savefig('img/maginales x')


#### Con curva de ajuste gaussiana
n = len(marginalesX) 
meanX = sum(xs*marginalesX)/n                   #note this correction
sigmaX = sum(marginalesX*(xs-meanX)**2)/n 

def gaus(x,a,x0,sigma):
       return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,xs,marginalesX,p0=[1,meanX,sigmaX])
plt.figure(3)
plt.plot(xs,marginalesX,'b:',label='data')
plt.plot(xs,gaus(xs,*popt),'r',label='fit')
plt.legend()
plt.savefig('img/maginales x con ajuste')


### (25 %) Asumir independencia de X y Y. Analíticamente, ¿cuál es entonces la 
###  expresión de la función de densidad conjunta que modela los datos?

"""
Como se sabe que 2 variables aleatorias son estadísticamente independientes si y solo si
P{X ≤ x, Y ≤ y} = P{X ≤ x}P{Y ≤ y} implica que FX,Y (x, y) = FX (x)FY (y) y además que 
fX,Y (x, y) = fX (x)fY (y); Por lo tanto:
FX (x|Y ≤ y) = FX (x)
"""

### (25 %) Hallar los valores de correlación, covarianza y coeficiente de correlación 
###  (Pearson) para los datos y explicar su significado.
lotes = pandas.read_csv('xyp.csv')

lotes_py = lotes.to_numpy()
lotes = np.array(lotes_py).astype("float")

X = lotes[:,0]
Y = lotes[:,1]
P = lotes[:,2]

#### Correlación:
relXY = 0

for c in range(len(X)):
    if c == 0:
        c += 1
    else:
        x_ = X[c]
        y_ = Y[c]
        prob = P[c]

        relXY += x_*y_*prob

print('\n   -> Correlación:     '+str(relXY))


#### Covarianza: 
cXY = relXY - (meanX*meanY)
print('\n   -> Covarianza:      '+str(cXY) )


#### Coeficiente de correlación:
ccr = cXY / (sigmaX*sigmaY)
print('\n   -> Coeficiente de correlación:      '+str(ccr) )


### (25 %) Graficar las funciones de densidad marginales (2D), la función de densidad 
###  conjunta (3D).
tresD = plt.figure(4)
ax = plt.axes(projection="3d")
ax.plot3D(X, Y, P, 'gray')
plt.savefig('img/3D')
plt.show()
