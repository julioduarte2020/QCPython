# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:20:09 2020

@author: jmduarte
"""

import pandas
import os

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('@')
    return tokenize[1]

def getKind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[1]

#TEMPERATURE
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura_raw/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura/'
L = os.listdir(root_in)

for i in range(len(L)):
    dictionary = {}
    key = 0
    file = root_in + L[i]
    print(L[i])
    estacion = getEstacion(L[i])
    kind = getKind(L[i])
    if kind == 'MEDIA': temp = 'mean'
    elif kind == 'MN': temp = 'min'
    else: temp = 'max'
    f = open(file, "r")
    s = f.readline()
    s = f.readline()
    while s != '':
        time = s[0:10]
        time = pandas.to_datetime(time)
        val = float(s[20:])
        dictionary[key] = {'codigo':estacion,
                           'time':time,
                           'vals':val,
                           'temp':temp}
        key += 1
        s = f.readline()
    f.close()
    table = pandas.DataFrame.from_dict(dictionary, 'index')
    if kind == 'MEDIA': output = '_mean_temperature.csv'
    elif kind == 'MN': output = '_min_temperature.csv'
    else: output = '_max_temperature.csv'
    table.to_csv(root_out + estacion + output)

#PRECIPITATION
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion_raw/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion/'
L = os.listdir(root_in)

for i in range(len(L)):
    dictionary = {}
    key = 0
    file = root_in + L[i]
    print(L[i])
    estacion = getEstacion(L[i])
    f = open(file, "r")
    s = f.readline()
    s = f.readline()
    while s != '':
        time = s[0:10]
        time = pandas.to_datetime(time)
        val = float(s[20:])
        dictionary[key] = {'codigo':estacion,
                           'time':time,
                           'vals':val}
        key += 1
        s = f.readline()
    f.close()
    table = pandas.DataFrame.from_dict(dictionary, 'index')
    table.to_csv(root_out + estacion + '_precipitation.csv')

#RELATIVE HUMIDITY
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Humedad_raw/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Humedad/'
L = os.listdir(root_in)

for i in range(len(L)):
    dictionary = {}
    key = 0
    file = root_in + L[i]
    print(L[i])
    estacion = getEstacion(L[i])
    f = open(file, "r")
    s = f.readline()
    s = f.readline()
    while s != '':
        time = s[0:19]
        time = pandas.to_datetime(time)
        val = float(s[20:])
        dictionary[key] = {'codigo':estacion,
                           'time':time,
                           'vals':val}
        key += 1
        s = f.readline()
    f.close()
    table = pandas.DataFrame.from_dict(dictionary, 'index')
    table.to_csv(root_out + estacion + '_humidity.csv')
  
 #WIND VELOCITY
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/VelocidadViento_raw/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/VelocidadViento/'
L = os.listdir(root_in)

for i in range(len(L)):
    dictionary = {}
    key = 0
    file = root_in + L[i]
    print(L[i])
    estacion = getEstacion(L[i])
    f = open(file, "r")
    s = f.readline()
    s = f.readline()
    while s != '':
        time = s[0:19]
        time = pandas.to_datetime(time)
        val = float(s[20:])
        dictionary[key] = {'codigo':estacion,
                           'time':time,
                           'vals':val}
        key += 1
        s = f.readline()
    f.close()
    table = pandas.DataFrame.from_dict(dictionary, 'index')
    table.to_csv(root_out + estacion + '_windVelocity.csv')   
    
    