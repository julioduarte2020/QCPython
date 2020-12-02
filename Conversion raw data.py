# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:53:52 2020

@author: jmduarte
"""

import pandas
import os

#TEMPERATURE
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/Temperatura_raw/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/Temperatura/'
L = os.listdir(root_in)

for i in range(len(L)):
    dictionary = {}
    key = 0
    file = root_in + L[i]
    print(L[i])
    f = open(file, "r")
    s = f.readline()
    while s != '':
        codigo = s[1:9]
        anio = s[11:15]
        dia = s[15:17]
        t = int(s[89:100])
        if t==1: temp = 'mean'
        elif t==2: temp = 'max'
        else: temp = 'min' 
        cont = 17
        for j in range(12):
            val = float(s[cont:cont+5])
            source = str(s[cont+5])
            cont += 6
            if j+1 < 10: mes = '0' + str(j+1)
            else: mes = str(j+1)
            time = anio + mes + dia
            if pandas.to_datetime(time, errors='coerce') is not pandas.NaT:
                dictionary[key] = {'codigo':codigo,
                                   'time':pandas.to_datetime(time),
                                   'vals':val,
                                   'source':source,
                                   'temp': temp}
                key += 1
        s = f.readline()         
    f.close()
    table = pandas.DataFrame.from_dict(dictionary, 'index')
    table = table.sort_values(by = ['codigo', 'time'])
    table.reset_index(drop=True, inplace=True)
    stations = table.codigo.unique()
    
    for j in range(len(stations)):
        subtable = table[table['codigo'] == stations[j]]
        subtable.reset_index(drop=True, inplace=True)
        meantable = subtable[subtable['temp'] == 'mean']
        if not(meantable.empty):
            meantable.reset_index(drop=True, inplace=True)
            meantable.to_csv(root_out + stations[j] + '_mean_temperature' + '.csv')
        maxtable = subtable[subtable['temp'] == 'max']
        if not(maxtable.empty):
            maxtable.reset_index(drop=True, inplace=True)
            maxtable.to_csv(root_out + stations[j] + '_max_temperature' + '.csv')
        mintable = subtable[subtable['temp'] == 'min']
        if not(mintable.empty):
            mintable.reset_index(drop=True, inplace=True)
            mintable.to_csv(root_out + stations[j] + '_min_temperature' + '.csv')
        
        
#PRECIPITATION
        
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/Precipitacion_raw/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/Precipitacion/'
L = os.listdir(root_in)

for i in range(len(L)):
    dictionary = {}
    key = 0
    file = root_in + L[i]
    print(L[i])
    f = open(file, "r")
    s = f.readline()
    while s != '':
        codigo = s[1:9]
        anio = s[11:15]
        dia = s[15:17]
        cont = 17
        for j in range(12):
            if s[cont:cont+5] == '     ': val = 99999
            else: val = float(s[cont:cont+5])
            source = str(s[cont+5])
            cont += 6
            if j+1 < 10: mes = '0' + str(j+1)
            else: mes = str(j+1)
            time = anio + mes + dia
            if pandas.to_datetime(time, errors='coerce') is not pandas.NaT:
                dictionary[key] = {'codigo':codigo,
                                   'time':pandas.to_datetime(time),
                                   'vals':val,
                                   'source':source}
                key += 1
        s = f.readline()         
    f.close()
    table = pandas.DataFrame.from_dict(dictionary, 'index')
    table = table.sort_values(by = ['codigo', 'time'])
    table.reset_index(drop=True, inplace=True)
    stations = table.codigo.unique()
    
    for j in range(len(stations)):
        subtable = table[table['codigo'] == stations[j]]
        subtable.reset_index(drop=True, inplace=True)
        subtable.to_csv(root_out + stations[j] + '_precipitation' + '.csv')