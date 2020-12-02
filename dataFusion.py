# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:51:49 2020

@author: jmduarte
"""

                                    ###############
                                    # TEMPERATURE #
                                    ###############

import os
import pandas

def get_station_kind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

root_in_1 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura/'
root_in_2 = 'D:/Escritorio/Corpoica/Calidad de datos/Temperatura0/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'

L1 = os.listdir(root_in_1)
L2 = os.listdir(root_in_2)

for i in range(len(L1)):
    (station1,kind1) = get_station_kind(L1[i])
    print(str(i) + ': ' +L1[i])
    table1 = pandas.read_csv(root_in_1 + L1[i], index_col=0)
    for j in range(len(L2)):
        (station2,kind2) = get_station_kind(L2[j])
        if station1 == station2 and kind1 == kind2:
            table2 = pandas.read_csv(root_in_2 + L2[j], index_col=0)
            table2.drop(columns=['source','Ideam','Duplicates'], inplace=True)
            for k in range(table1.shape[0]):
                date = table1.loc[k,'time']
                subtable2 = table2[table2['time'] == date]
                if subtable2.empty: 
                    subtable1 = table1[table1['time'] == date]
                    table2 = pandas.concat([table2,subtable1], ignore_index=True)
                else: table2.loc[subtable2.index,'vals'] = table1.loc[k,'vals']
            table2.to_csv(root_out + L2[j])    
            

                                    #################
                                    # PRECIPITATION #
                                    #################


import os
import pandas

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

root_in_1 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion/'
root_in_2 = 'D:/Escritorio/Corpoica/Calidad de datos/Precipitacion0/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'

L1 = os.listdir(root_in_1)
L2 = os.listdir(root_in_2)

for i in range(len(L1)):
    station1 = get_station(L1[i])
    print(str(i) + ': ' +L1[i])
    table1 = pandas.read_csv(root_in_1 + L1[i], index_col=0)
    for j in range(len(L2)):
        station2 = get_station(L2[j])
        if station1 == station2:
            table2 = pandas.read_csv(root_in_2 + L2[j], index_col=0)
            table2.drop(columns=['source','Ideam','Error Acumulado','Duplicates'], inplace=True)
            for k in range(table1.shape[0]):
                date = table1.loc[k,'time']
                subtable2 = table2[table2['time'] == date]
                if subtable2.empty: 
                    subtable1 = table1[table1['time'] == date]
                    table2 = pandas.concat([table2,subtable1], ignore_index=True)
                else: table2.loc[subtable2.index,'vals'] = table1.loc[k,'vals']
            table2.to_csv(root_out + L2[j])    
            
