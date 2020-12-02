# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:42:09 2020

@author: jmduarte
"""
###############
# TEMPERATURE #
###############
import pandas
import os
from pyproj import Proj
import math
import numpy
from pykrige.ok3d import OrdinaryKriging3D

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura2/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura3/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
project = Proj(proj='utm',zone=18,ellps='WGS84')
for i in range(len(L)):
    (estacion,tipo) = getEstacion(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Filling',0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == estacion]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = project(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        coords = []
        for j in range(len(L)):
            if i != j:
                (estacion2,tipo2) = getEstacion(L[j])
                if tipo == tipo2:
                    row2 = CNE[CNE['CODIGO'] == estacion2]
                    if not(row2.empty):
                        row2.reset_index(drop=True, inplace=True)
                        lat2 = row2.loc[0,'latitud']
                        long2 = row2.loc[0,'longitud']
                        x2,y2 = project(long2, lat2)
                        H2 = row2.loc[0,'altitud']
                        if abs(H-H2) <= 500:
                            D = math.sqrt((x-x2)**2 + (y-y2)**2)
                            if D <= 200000: 
                                neighbors.append(L[j])
                                coords.append((x2,y2,H2))
        if len(neighbors) >= 3:
            neighbors_list = []
            for j in range(len(neighbors)):
                neighbors_list.append( pandas.read_csv(root_in + neighbors[j], index_col=0) )
                neighbors_list[j]['time'] = pandas.to_datetime(neighbors_list[j]['time'])
            Length = table.shape[0]
            for k in range(Length):
                val = table.loc[k,'vals']
                if val < 99999.0: continue
                date = table.loc[k,'time']
                x_n = []
                y_n = []
                H_n = []
                val_n = []
                for n in range(len(neighbors_list)):
                    row = neighbors_list[n][neighbors_list[n]['time'] == date]
                    if row.empty: continue
                    row.reset_index(drop=True, inplace=True)
                    val2 = row.loc[0,'vals']
                    if val2 >= 99999.0: continue
                    x_n.append(coords[n][0])
                    y_n.append(coords[n][1])
                    H_n.append(coords[n][2])
                    val_n.append(val2)
                if len(x_n) < 3: continue
                x_n = numpy.asarray(x_n)
                y_n = numpy.asarray(y_n)
                H_n = numpy.asarray(H_n)
                val_n = numpy.asarray(val_n)
                if numpy.std(val_n)>0.01:
                    try:
                        ok3d = OrdinaryKriging3D(x_n, y_n, H_n, val_n, variogram_model="linear")
                        val_hat,ss3d = ok3d.execute("points", x, y, H)
                        if val_hat.data[0] < 0.0 or val_hat.data[0] > 50: continue
                        table.loc[k,'vals'] = round(val_hat.data[0],1)
                    except: continue
                else: table.loc[k,'vals'] = numpy.median(val_n)
                table.loc[k,'Filling'] = 1
    table.to_csv(root_out + L[i])                        

#################
# PRECIPITATION #
#################
import pandas
import os
from pyproj import Proj
import math
import numpy
#from scipy.stats import wilcoxon
from pykrige.ok3d import OrdinaryKriging3D

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion2/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion3/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
project = Proj(proj='utm',zone=18,ellps='WGS84')
for i in range(len(L)):
    estacion = getEstacion(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Filling',0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == estacion]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = project(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        coords = []
        for j in range(len(L)):
            if i != j:
                estacion2 = getEstacion(L[j])
                row2 = CNE[CNE['CODIGO'] == estacion2]
                if not(row2.empty):
                    row2.reset_index(drop=True, inplace=True)
                    lat2 = row2.loc[0,'latitud']
                    long2 = row2.loc[0,'longitud']
                    x2,y2 = project(long2, lat2)
                    H2 = row2.loc[0,'altitud']
                    D = math.sqrt((x-x2)**2 + (y-y2)**2)
                    if D <= 100000: 
                        neighbors.append(L[j])
                        coords.append((x2,y2,H2))
        if len(neighbors) >= 3:
            neighbors_list = []
            for j in range(len(neighbors)):
                neighbors_list.append( pandas.read_csv(root_in + neighbors[j], index_col=0) )
                neighbors_list[j]['time'] = pandas.to_datetime(neighbors_list[j]['time'])
            Length = table.shape[0]
            for k in range(Length):
                val = table.loc[k,'vals']
                if val < 99999.0: continue
                date = table.loc[k,'time']
                x_n = []
                y_n = []
                H_n = []
                val_n = []
                for n in range(len(neighbors_list)):
                    row = neighbors_list[n][neighbors_list[n]['time'] == date]
                    if row.empty: continue
                    row.reset_index(drop=True, inplace=True)
                    val2 = row.loc[0,'vals']
                    if val2 >= 99999.0: continue
                    x_n.append(coords[n][0])
                    y_n.append(coords[n][1])
                    H_n.append(coords[n][2])
                    val_n.append(val2)
                if len(x_n) < 3: continue
                x_n = numpy.asarray(x_n)
                y_n = numpy.asarray(y_n)
                H_n = numpy.asarray(H_n)
                val_n = numpy.asarray(val_n)
                if numpy.std(val_n)>0.01:
                    try:
                        ok3d = OrdinaryKriging3D(x_n, y_n, H_n, val_n, variogram_model="linear")
                        val_hat,ss3d = ok3d.execute("points", x, y, H)
                        if math.copysign(1,val_hat.data[0]) >= 0.0: 
                            if val_hat.data[0] > 200: continue
                            table.loc[k,'vals'] = round(val_hat.data[0],1)
                        else: table.loc[k,'vals'] = 0.0
                    except: continue
                else: table.loc[k,'vals'] = numpy.median(val_n)
                table.loc[k,'Filling'] = 1
    table.to_csv(root_out + L[i])                        