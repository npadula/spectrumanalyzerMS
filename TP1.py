# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:49:32 2016

@author: npadula
"""

import numpy
import scipy
from scipy import signal,fftpack, pi
from matplotlib import pylab

#Funciones auxiliares

#numpy.where() devuelve los INDICES de los elementos del array
#que cumplan las condiciones, getValores se usa para obtener los
#elementos en si a partir de dichos indices
def getValores(indices, a):
    resultado = numpy.array([])
    for i in numpy.nditer(indices):
        tmp = numpy.array(a[i])
        resultado = numpy.hstack((resultado,tmp))
    
    return resultado
    

#Programa principal

cantidadDeMuestras = 4000
espacioEntreMuestras = 1.0 / 4000

#Puntos en el tiempo
t = numpy.linspace(0.0, cantidadDeMuestras*espacioEntreMuestras, cantidadDeMuestras)

#Funciones para esos puntos en t
#y = numpy.sin(50.0 * 2.0*numpy.pi*t) + 0.5*numpy.sin(80.0 * 2.0*numpy.pi*t)
#y = signal.square(200*2.0*pi*t)
#y = signal.square(2.0*pi*t)
#y = scipy.signal.sawtooth(100 *2.0*pi*t, 0.5)
#y = scipy.signal.sawtooth(200 *2.0*pi*t) 
y = scipy.sin(1100 *2*pi*t) 

yf = fftpack.fft(y) #Calculo de la FFT
#Puntos en frecuencia asociados a yf
tf = fftpack.fftfreq(len(yf),espacioEntreMuestras)
tf = tf[0:cantidadDeMuestras/2]
#tf = numpy.linspace(200.0, 1.0/(2.0*espacioEntreMuestras), cantidadDeMuestras/2)


#Ploteo
pylab.subplot(221)
pylab.title('Transformada de Fourier Y(f)') 
pylab.xlabel('Frecuencia [hz]')
pylab.ylabel('Magnitud |Y(f)|')
pylab.plot(tf,numpy.abs(yf[0:cantidadDeMuestras/2]))  



#Particion de los intervalos a integrar correspondientes a las bandas
#del analizador espectral
#TODO: refactorizar esto!, codigo por demas redundante
offset = 0

#200 a 560hz
intervaloTf1 = getValores((numpy.where(tf <= 560))[0],tf)
intervaloYf1 = yf[offset:len(intervaloTf1)]
offset += len(intervaloTf1)

#560 a 920hz
intervaloTf2 = getValores((numpy.where((tf > 560) & (tf <= 920) ) )[0],tf)
intervaloYf2 = yf[offset:offset + len(intervaloTf2)]
offset += len(intervaloTf2)

#920a 1280hz
intervaloTf3 = getValores((numpy.where((tf > 920) & (tf <= 1280) ) )[0],tf)
intervaloYf3 = yf[offset:offset + len(intervaloTf3)]
offset += len(intervaloTf3)

#1280 a 1640hz
intervaloTf4 = getValores((numpy.where((tf > 1280) & (tf <= 1640) ) )[0],tf)
intervaloYf4 = yf[offset:offset + len(intervaloTf4)]
offset += len(intervaloTf4)

#1640 a 2000hz
intervaloTf5 = getValores((numpy.where((tf > 1640) & (tf <= 2000) ) )[0],tf)
intervaloYf5 = yf[offset:offset + len(intervaloTf5)]
offset += len(intervaloTf5)

#Integración numérica de cada intervalo, el resultado de cada integral es
#la magnitud de cada banda del analizador espectral

banda1 = scipy.integrate.trapz(numpy.abs(intervaloYf1)**2,intervaloTf1,espacioEntreMuestras)
banda2 = scipy.integrate.trapz(numpy.abs(intervaloYf2)**2,intervaloTf2,espacioEntreMuestras)
banda3 = scipy.integrate.trapz(numpy.abs(intervaloYf3)**2,intervaloTf3,espacioEntreMuestras)
banda4 = scipy.integrate.trapz(numpy.abs(intervaloYf4)**2,intervaloTf4,espacioEntreMuestras)
banda5 = scipy.integrate.trapz(numpy.abs(intervaloYf5)**2,intervaloTf5,espacioEntreMuestras)



#Plotea el grafico de barras del analizador
energiaBase = 1 #Simula el ruido termico de un circuito, evita que el calculo de dB de valores < 0
ejeFrecuencias = ['200','560','920','1280','1640','2000'] #Cortes de cada banda
ejeMagnitud = numpy.array([banda1,banda2,banda3,banda4,banda5, 0]) #Magnitud de cada banda, la ultima es cero para que el 2000hz quede bien al final
ejeMagnitud = 20*numpy.log10(energiaBase + ejeMagnitud) #Convierte las magnitudes calculadas a dB
ejeMagnitud[5] = 0.000000001 #es cero para que el 2000hz quede bien al final. Debería haber una forma mas decente de hacer esto

pylab.subplot(222)
pylab.xlabel('Frecuencia [hz]')
pylab.ylabel('Energia de Y(f) [dB]')
pylab.title('Analizador espectral')
pylab.xticks(numpy.arange(len(ejeFrecuencias)),ejeFrecuencias)
pylab.bar(numpy.arange(len(ejeFrecuencias)),ejeMagnitud, alpha=0.5, color='cyan', edgecolor='blue')


pylab.show()


        




