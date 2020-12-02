# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:30:36 2020

@author: jmduarte
"""

import pandas
import os

def average_duplicates(table, bool_series):
    cont = 0
    while cont < table.shape[1]:
        if bool_series[cont]:
            if table.loc[cont, 'vals'] < 99999.0 and table.loc[cont+1, 'vals'] < 99999.0:
                mean = 0.5 * (table.loc[cont, 'vals'] + table.loc[cont+1, 'vals'])
                table.loc[cont, 'vals'] = mean
                table.loc[cont+1, 'vals'] = mean
            elif table.loc[cont, 'vals'] < 99999.0:
                table.loc[cont+1, 'vals'] = table.loc[cont, 'vals']
            elif table.loc[cont+1, 'vals'] < 99999.0:
                table.loc[cont, 'vals'] = table.loc[cont+1, 'vals']
            cont += 2
        else: cont += 1

#############
#TEMPERATURE#
#############
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/Temperatura/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/Temperatura0/'
root_reports = 'D:/Escritorio/Corpoica/Calidad de datos/'
L = os.listdir(root_in)

report = pandas.DataFrame(columns = ['codigo','time','vals','source','temp','Ideam','Duplicates'])

for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0, dtype={'source': str})
    table.insert(table.shape[1],'Ideam',0)
    table.insert(table.shape[1],'Duplicates',0)
    print(L[i])
    #Repeated dates
    bool_series = table['time'].duplicated(keep=False)
    if bool_series.sum() > 0:
        table.loc[bool_series,['Duplicates']] = 1 
        average_duplicates(table, bool_series)
        bool_series = table['time'].duplicated(keep='first')
        table = table[~bool_series]
        table.reset_index(drop=True, inplace=True)
        
    #Error tipo 1
    subtable = table[(table['vals'] == 99999.0) & (table['source'] != '0')]
    if not(subtable.empty):
        table.loc[subtable.index,['source']] = '0'
        table.loc[subtable.index,['Ideam']] = 1
        
    #Error tipo 2
    cond1 = table['source'] != '0'
    cond2 = table['source'] != '1'
    cond3 = table['source'] != '2'
    cond = cond1 & cond2 & cond3 #source different from 0,1,2
    subtable = table[(table['vals'] < 99999.0) & cond]
    if not(subtable.empty):
        table.loc[subtable.index,['source']] = '2'
        table.loc[subtable.index,['Ideam']] = 2
    
    #Fill the report with the errors found
    cond1 = table['Ideam'] > 0
    cond2 = table['Duplicates'] > 0 
    subtable = table[cond1 | cond2]
    if not(subtable.empty):
        report = pandas.concat([report,subtable],ignore_index=True)    
        
    table.to_csv(root_out + L[i])

report.drop(columns=['vals','source'], inplace=True) 
report.to_csv(root_reports + 'Report_Temperature0.csv')    
 
###############
#PRECIPITATION#
###############
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/Precipitacion/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/Precipitacion0/'
root_reports = 'D:/Escritorio/Corpoica/Calidad de datos/'
L = os.listdir(root_in)

report = pandas.DataFrame(columns = ['codigo','time','vals',
                                     'source','Ideam','Error Acumulado','Duplicates'])
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0, dtype={'source': str})
    table.insert(table.shape[1],'Ideam',0)
    table.insert(table.shape[1],'Error Acumulado',0)
    table.insert(table.shape[1],'Duplicates',0)
    print(L[i])
    #Repeated dates
    bool_series = table['time'].duplicated(keep=False)
    if bool_series.sum() > 0:
        table.loc[bool_series,['Duplicates']] = 1 
        average_duplicates(table, bool_series)
        bool_series = table['time'].duplicated(keep='first')
        table = table[~bool_series]
        table.reset_index(drop=True, inplace=True)
    #Error tipo 1
    cond1 = table['source'] != '0'
    cond2 = table['source'] != 'B'
    cond = cond1 & cond2
    subtable = table[(table['vals'] == 99999.0) & cond]
    if not(subtable.empty):
        table.loc[subtable.index,['source']] = '0'
        table.loc[subtable.index,['Ideam']] = 1
    #Error tipo 2
    subtable = table[(table['vals'] < 99999.0) & (table['source'] == 'B')]
    if not(subtable.empty):
        table.loc[subtable.index,['vals']] = 99999.0
        table.loc[subtable.index,['Ideam']] = 2
    #Error tipo 3
    cond1 = table['source'] != '0'
    cond2 = table['source'] != '1'
    cond3 = table['source'] != '2'
    cond4 = table['source'] != ':'
    cond = cond1 & cond2 & cond3 & cond4 #source different from 0,1,2,:
    subtable = table[(table['vals'] < 99999.0) & cond]
    if not(subtable.empty):
        table.loc[subtable.index,['source']] = '2'
        table.loc[subtable.index,['Ideam']] = 3

    #Distribute valid accumulations
    cond = table['source'] == 'B'
    Lb = len(cond)
    for j in range(0,Lb):
        #Block size is one -B-
        if j == 0 and cond[0] and not(cond[1]):
            if table.loc[1,'vals']>=1.0 and (table.loc[1,'source']=='1' or table.loc[1,'source']=='2'):
                table.loc[0,'vals'] = table.loc[1,'vals'] / 2
                table.loc[0,'source'] = ':'
                table.loc[1,'vals'] = table.loc[1,'vals'] / 2
                table.loc[1,'source'] = ':'
            else: 
                table.loc[1,'vals'] = 99999.0
                table.loc[1,'source'] = '0'
                table.loc[1,'Error Acumulado'] = 1
        if 0<j<Lb-1 and cond[j] and not(cond[j-1]) and not(cond[j+1]):
            if (table.loc[j+1,'vals']>=1.0 and (table.loc[j+1,'source']=='1' or table.loc[j+1,'source']=='2')):
                table.loc[j,'vals'] = table.loc[j+1,'vals'] / 2
                table.loc[j,'source'] = ':'
                table.loc[j+1,'vals'] = table.loc[j+1,'vals'] / 2
                table.loc[j+1,'source'] = ':'
            else:
                table.loc[j+1,'vals'] = 99999.0
                table.loc[j+1,'source'] = '0'
                table.loc[j+1,'Error Acumulado'] = 1
        #Block size is two -BB-
        if j == 1 and cond[1] and cond[0] and not(cond[2]):
            if (table.loc[2,'vals']>=1.0) and (table.loc[2,'source']=='1' or table.loc[2,'source']=='2'):
                table.loc[0,'vals'] = table.loc[2,'vals'] / 3
                table.loc[0,'source'] = ':'
                table.loc[1,'vals'] = table.loc[2,'vals'] / 3
                table.loc[1,'source'] = ':'
                table.loc[2,'vals'] = table.loc[2,'vals'] / 3
                table.loc[2,'source'] = ':'
            else:
                table.loc[2,'vals'] = 99999.0
                table.loc[2,'source'] = '0'
                table.loc[2,'Error Acumulado'] = 1
        if 1<j<Lb-1 and cond[j] and cond[j-1] and not(cond[j+1]) and not(cond[j-2]):
          if (table.loc[j+1,'vals']>=1.0) and (table.loc[j+1,'source']=='1' or table.loc[j+1,'source']=='2'):
                table.loc[j-1,'vals'] = table.loc[j+1,'vals'] / 3
                table.loc[j-1,'source'] = ':'
                table.loc[j,'vals'] = table.loc[j+1,'vals'] / 3
                table.loc[j,'source'] = ':'
                table.loc[j+1,'vals'] = table.loc[j+1,'vals'] / 3
                table.loc[j+1,'source'] = ':'
          else:
                table.loc[j+1,'vals'] = 99999.0
                table.loc[j+1,'source'] = '0'
                table.loc[j+1,'Error Acumulado'] = 1
    
    #Fill the report with the errors found
    cond1 = table['Ideam'] > 0
    cond2 = table['Duplicates'] > 0 
    cond3 = table['Error Acumulado'] > 0
    cond4 = table['source'] == ':'
    subtable = table[cond1 | cond2 | cond3 | cond4]
    if not(subtable.empty):
        report = pandas.concat([report,subtable],ignore_index=True)    

    table.to_csv(root_out + L[i])
    
report.drop(columns=['vals'], inplace=True) 
report.to_csv(root_reports + 'Report_Precipitation0.csv')    
   