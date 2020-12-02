# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:23:39 2020

@author: jmduarte
"""


                                    ###############
                                    # TEMPERATURE #
                                    ###############

import pandas
from pandas.tseries.offsets import DateOffset
import os
import numpy
from statsmodels import robust
import math

def weights(x, M, MAD):
    c = 7.5
    den = c*MAD
    Lw = x.size
    u = numpy.zeros(Lw)
    for k in range(Lw):
        u[k] = x[k] - M
    u /= den 
    for k in range(Lw):
        if abs(u[k]) > 1.0: u[k] = 1.0
    return u

def mean_biweight(x, M, u):
    Lw = x.size
    num = 0
    den = 0
    for k in range(Lw):
        w = 1.0 - u[k]*u[k]
        num += (x[k] - M) * w * w
        den += w * w
    x_mean = M + num/den
    return x_mean    
     
def std_biweight(x, M, u):
    Lw = x.size
    num = 0
    den = 0
    for k in range(Lw):
        w = 1.0 - u[k]*u[k]
        w2 = 1.0 - 5*u[k]*u[k]
        num += pow(x[k] - M, 2) * pow(w, 4)
        den += w * w2
    num *= Lw
    num = math.sqrt(num)
    den = abs(den)
    x_std = num / den
    return x_std 
    
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

L = os.listdir(root_in)

###################################################################
# EXTREMES, DESVIACIONES RESPECTO AL RANGO INTERQUARTIL, BIWEIGHT #
###################################################################
Tmax = 50
Tmin = -10
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Extremes',0)
    table.insert(table.shape[1],'DRI',0)
    table.insert(table.shape[1],'biweigth',0)
    print(str(i) + ': ' +L[i])
    ##########
    #EXTREMES#
    ##########
    cond1 = table['vals'] < 99999.0
    cond2 = (table['vals'] < Tmin) | (table['vals'] > Tmax)
    cond = cond1 & cond2
    if cond.sum() > 0: table.loc[cond,['Extremes']] = 1
    #############################################################
    #DESVIACION RESPECTO AL RANGO INTERQUARTIL (DRI) y BIWEIGHT #
    #############################################################
    table['time'] = pandas.to_datetime(table['time'])
    first_date = table.loc[0,'time']
    last_date = table.loc[table.shape[0]-1,'time']
    if first_date.is_leap_year: last_date_first_year = first_date + DateOffset(days=365)
    else: last_date_first_year = first_date + DateOffset(days=364)
    current_date = first_date
    while current_date <= last_date_first_year:
        W = []
        if current_date < first_date + DateOffset(days=2):
            current_date_plus_years = first_date
            while current_date_plus_years <= last_date:
                for k in range(5):
                    row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                    if not(row.empty):
                        row.reset_index(drop=True, inplace=True)
                        value = row.loc[0,'vals']
                        if value < 99999.0: W.append(value)
                current_date_plus_years += DateOffset(years=1)        
        elif current_date >= first_date + DateOffset(days=2) and \
            current_date <= last_date_first_year - DateOffset(days=2):
            current_date_plus_years = current_date
            while current_date_plus_years <= last_date:
                for k in range(-2,3):
                    row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                    if not(row.empty):
                        row.reset_index(drop=True, inplace=True)
                        value = row.loc[0,'vals']
                        if value < 99999.0: W.append(value)
                current_date_plus_years += DateOffset(years=1)
        else:
            current_date_plus_years = last_date_first_year - DateOffset(days=4)
            while current_date_plus_years <= last_date:
                for k in range(5):
                    row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                    if not(row.empty):
                        row.reset_index(drop=True, inplace=True)
                        value = row.loc[0,'vals']
                        if value < 99999.0: W.append(value)
                current_date_plus_years += DateOffset(years=1)
        W = numpy.asarray(W)
        if W.size >= 3:
            median = numpy.median(W)
            ################DRI###############
            ir = numpy.percentile(W,75) -  numpy.percentile(W,25) #Interquartil range
            ssd = ir / 1.349 #pseudo standard deviation
            ##############BIWEIGTH###########
            MAD = robust.mad(W)
            if MAD > 0:
                u = weights(W, median, MAD)
                mean = mean_biweight(W, median, u)
                std = std_biweight(W, median, u)
            else:
                mean = median
                std = 0
            #################################
            current_date_plus_years = current_date
            while current_date_plus_years <= last_date:
                row = table[table['time'] == current_date_plus_years]
                if not(row.empty):
                    index = row.index
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals'] 
                    if value < 99999.0:
                        if ssd > 0:
                            Z = abs(value - median) / ssd
                            if Z > 3: table.loc[index,'DRI'] = 1
                        if std > 0:
                            Z = abs(value - mean) / std
                            if Z > 3: table.loc[index,'biweigth'] = 1
                current_date_plus_years += DateOffset(years=1)
        current_date += DateOffset(days=1)
    
    table.to_csv(root_out + L[i])
    
###################################################
# DESVIACIONES RESPECTO AL CICLO ESTACIONAL (DCE) #
###################################################
import pandas
from patsy import dmatrix
import statsmodels.api as sm
import os
import numpy

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'DCE',0)
    print(str(i) + ': ' +L[i])
    table['time'] = pandas.to_datetime(table['time'])
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    current_year = first_year
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if not(subtable.empty):
            subtable = subtable[subtable['vals'] < 99999.0]
            if subtable.shape[0] < 100: 
                current_year += 1
                continue
            x = pandas.DatetimeIndex(subtable['time']).dayofyear
            x = numpy.asarray(x)
            y = subtable['vals'].to_numpy()
            x_min = numpy.min(x)
            x_max = numpy.max(x)
            kts = numpy.linspace(x_min,x_max,10)
            knots = 'knots=('
            for k in range(10):
                knots += str(kts[k])
                if k<9: knots += ','
            knots += '),'
            try:    
                cubic = dmatrix('bs(data,' + knots + 'include_intercept = False)', 
                                {'data': x}, return_type = 'dataframe')
                model = sm.GLM(y, cubic).fit()
                y_hat = model.predict(cubic)
                y_hat = y_hat.to_numpy()
                desviacion = y - y_hat
                desv_99 = numpy.percentile(desviacion,99)
                desv_01 = numpy.percentile(desviacion,1)
                boolean = (desviacion < desv_01) | (desviacion > desv_99) 
                if boolean.sum() > 0:
                    for j in range(boolean.shape[0]):
                        if boolean[j]: table.loc[subtable.index[j],'DCE'] = 1
            except:
                current_year += 1
                continue
        current_year += 1    
    table.to_csv(root_out + L[i])

###################################
#CONTROLES DE CONTINUIDAD TEMPORAL#
###################################
import pandas
import os
import numpy
from pandas.tseries.offsets import DateOffset

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    table.insert(table.shape[1],'Persistence',0)
    table.insert(table.shape[1],'Excesive Jumps',0)
    table.insert(table.shape[1],'Peaks',0)
    #Persistencia de tres o más días consecutivos
    j=0
    while j<table.shape[0]-2:
        if table.loc[j,'vals'] == 99999.0:
            j += 1
            continue
        W = [j]
        while table.loc[j,'vals'] == table.loc[j+1,'vals']:
            W.append(j+1)
            j += 1
            if j >= table.shape[0]-1: break
        Lw = len(W)
        if Lw >= 3:
            for k in range(Lw):
                table.loc[W[k],'Persistence']=1
        j += 1
    #Saltos excesivos entre días consecutivos
    table['time'] = pandas.to_datetime(table['time'])
    diff = []
    index = []
    for j in range(1,table.shape[0]):
        if table.loc[j,'vals']<99999.0 and table.loc[j-1,'vals']<99999.0:
            current_date = table.loc[j,'time']
            previous_date = current_date - DateOffset(days=1)
            if table.loc[j-1,'time'] == previous_date:
                diff.append( abs(table.loc[j,'vals'] - table.loc[j-1,'vals']) )
                index.append(j)
    diff = numpy.asarray(diff)
    if diff.size >= 3:
        Perc = numpy.percentile(diff,99)
        for j in range(diff.size):
            if diff[j] > Perc: table.loc[index[j],'Excesive Jumps'] = 1
        #Picos de corta duración
        for j in range(1,diff.size-1):
            current_date = table.loc[index[j],'time']
            previous_date = current_date - DateOffset(days=1)
            next_date = current_date + DateOffset(days=1)
            cond1 = table.loc[index[j-1],'time'] == previous_date
            cond2 = table.loc[index[j+1],'time'] == next_date 
            if cond1 and cond2:
               if diff[j] > Perc and diff[j+1] > Perc: table.loc[index[j],'Peaks'] = 1
    
    table.to_csv(root_out + L[i])

###################################
# CONSISTENCIA ENTRE TEMPERATURAS #
###################################
import pandas
import os
import numpy
from pandas.tseries.offsets import DateOffset

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

L = os.listdir(root_in)
i = 0
while i < len(L):
    estacion = get_station(L[i])
    print(str(i) + ': ' +estacion)
    max_file = estacion + '_max_temperature.csv'
    mean_file = estacion + '_mean_temperature.csv'
    min_file = estacion + '_min_temperature.csv'
    table_max = pandas.DataFrame()
    table_mean = pandas.DataFrame()
    table_min = pandas.DataFrame()
    if max_file == L[i]: 
        table_max = pandas.read_csv(root_in + L[i], index_col=0)
        i += 1
        if mean_file == L[i]:
            table_mean = pandas.read_csv(root_in + L[i], index_col=0)
            i += 1
            if min_file == L[i]:
                table_min = pandas.read_csv(root_in + L[i], index_col=0)
                i += 1
        elif min_file == L[i]:
            table_min = pandas.read_csv(root_in + L[i], index_col=0)
            i += 1
    elif mean_file == L[i]:
        table_mean = pandas.read_csv(root_in + L[i], index_col=0)
        i += 1
        if min_file == L[i]:
            table_min = pandas.read_csv(root_in + L[i], index_col=0)
            i += 1
    elif min_file == L[i]:
        table_min = pandas.read_csv(root_in + L[i], index_col=0)
        i += 1
    if not(table_max.empty): 
       table_max['time'] = pandas.to_datetime(table_max['time'])
       table_max.insert(table_max.shape[1],'Consistency 1',0)
       table_max.insert(table_max.shape[1],'Consistency 2',0)
       table_max.insert(table_max.shape[1],'Consistency 3',0)
       table_max.insert(table_max.shape[1],'Consistency 4',0)
       table_max.insert(table_max.shape[1],'Amplitud Termica',0)
       
    if not(table_mean.empty):    
       table_mean['time'] = pandas.to_datetime(table_mean['time'])
       table_mean.insert(table_mean.shape[1],'Consistency 1',0)
       table_mean.insert(table_mean.shape[1],'Consistency 2',0)
       table_mean.insert(table_mean.shape[1],'Consistency 3',0)
       table_mean.insert(table_mean.shape[1],'Consistency 4',0)
       table_mean.insert(table_mean.shape[1],'Amplitud Termica',0)
       
    if not(table_min.empty):
       table_min['time'] = pandas.to_datetime(table_min['time'])
       table_min.insert(table_min.shape[1],'Consistency 1',0)
       table_min.insert(table_min.shape[1],'Consistency 2',0)
       table_min.insert(table_min.shape[1],'Consistency 3',0)
       table_min.insert(table_min.shape[1],'Consistency 4',0)
       table_min.insert(table_min.shape[1],'Amplitud Termica',0)
    
    if not(table_max.empty) and not(table_mean.empty) and not(table_min.empty):         
       diff = []
       ind_max = []
       ind_mean = []
       ind_min = []
       Length = table_mean.shape[0]
       for j in range(Length):
           date = table_mean.loc[j,'time']
           T_mean = table_mean.loc[j,'vals']
           if T_mean == 99999.0: continue
           row_max = table_max[table_max['time'] == date]
           if not(row_max.empty):
               index_max = row_max.index
               row_max.reset_index(drop=True, inplace=True)
               T_max = row_max.loc[0,'vals']
               if T_max == 99999.0: continue
           else: continue
           row_min = table_min[table_min['time'] == date]
           if not(row_min.empty):
               index_min = row_min.index
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
           else: continue
           # Consistency 1
           if not( T_min < T_mean < T_max ):
               table_mean.loc[j,'Consistency 1'] = 1
               table_max.loc[index_max,'Consistency 1'] = 1
               table_min.loc[index_min,'Consistency 1'] = 1
           # Consistency 2
           T_av = 0.5 * (T_min + T_max)
           diff.append(abs(T_mean - T_av))
           ind_max.append(index_max)
           ind_mean.append(j)
           ind_min.append(index_min)
                   
       diff = numpy.asarray(diff)
       if diff.size >= 5:
           Thr = numpy.percentile(diff, 99.9)
           for j in range(diff.size):
               if diff[j] > Thr:
                   table_max.loc[ind_max[j],'Consistency 2'] = 1
                   table_mean.loc[ind_mean[j],'Consistency 2'] = 1
                   table_min.loc[ind_min[j],'Consistency 2'] = 1
    
    #Special cases
    if table_max.empty and not(table_mean.empty) and not(table_min.empty): 
       Length = table_mean.shape[0]
       for j in range(Length):
           date = table_mean.loc[j,'time']
           T_mean = table_mean.loc[j,'vals']
           if T_mean == 99999.0: continue
           row_min = table_min[table_min['time'] == date]
           if not(row_min.empty):
               index_min = row_min.index
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
           else: continue
           # Consistency 1
           if not( T_min < T_mean ):
               table_mean.loc[j,'Consistency 1'] = 1
               table_min.loc[index_min,'Consistency 1'] = 1
    if not(table_max.empty) and not(table_mean.empty) and table_min.empty:
       Length = table_mean.shape[0]
       for j in range(Length):
           date = table_mean.loc[j,'time']
           T_mean = table_mean.loc[j,'vals']
           if T_mean == 99999.0: continue
           row_max = table_max[table_max['time'] == date]
           if not(row_max.empty):
               index_max = row_max.index
               row_max.reset_index(drop=True, inplace=True)
               T_max = row_max.loc[0,'vals']
               if T_max == 99999.0: continue
           else: continue
           # Consistency 1
           if not( T_mean < T_max ):
               table_mean.loc[j,'Consistency 1'] = 1
               table_max.loc[index_max,'Consistency 1'] = 1
                
    # Consistencies 3 and 4
    if not(table_max.empty) and not(table_min.empty):
        Length = table_max.shape[0]
        for j in range(1,Length-2):
           T_max = table_max.loc[j,'vals']
           if T_max == 99999.0: continue 
           date = table_max.loc[j,'time']
           row_min = table_min[table_min['time'] == date]
           if not(row_min.empty):
               index_min = row_min.index
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
           else: continue
           date_before = date - DateOffset(days=1)
           date_after = date + DateOffset(days=1)
           row_min_before = table_min[table_min['time'] == date_before]
           row_min_after = table_min[table_min['time'] == date_after]
           if not(row_min_before.empty) and not(row_min_after.empty):
               index_min_before = row_min_before.index
               row_min_before.reset_index(drop=True, inplace=True)
               T_min_before = row_min_before.loc[0,'vals']
               index_min_after = row_min_after.index
               row_min_after.reset_index(drop=True, inplace=True)
               T_min_after = row_min_after.loc[0,'vals']
               if T_min_before < 99999.0 and T_min_after < 99999.0:
                   if not(T_min_before <= T_max >= T_min_after):
                       table_max.loc[j,'Consistency 3'] = 1
                       table_min.loc[index_min_before,'Consistency 3'] = 1
                       table_min.loc[index_min_after,'Consistency 3'] = 1
           row_max_before = table_max[table_max['time'] == date_before]
           row_max_after = table_max[table_max['time'] == date_after]
           if not(row_max_before.empty) and not(row_max_after.empty):
               index_max_before = row_max_before.index
               row_max_before.reset_index(drop=True, inplace=True)
               T_max_before = row_max_before.loc[0,'vals']
               index_max_after = row_max_after.index
               row_max_after.reset_index(drop=True, inplace=True)
               T_max_after = row_max_after.loc[0,'vals']
               if T_max_before < 99999.0 and T_max_after < 99999.0:
                   if not(T_max_before >= T_min <= T_max_after):
                       table_min.loc[index_min,'Consistency 4'] = 1
                       table_max.loc[index_max_before,'Consistency 4'] = 1
                       table_max.loc[index_max_after,'Consistency 4'] = 1
           #Amplitud Térmica
           Amplitud_termica = T_max - T_min
           if not( 0.01 <= Amplitud_termica <= 30 ):
               table_max.loc[j,'Amplitud Termica'] = 1
               table_min.loc[index_min,'Amplitud Termica'] = 1
               
    if not(table_max.empty): table_max.to_csv(root_out + max_file)
    if not(table_mean.empty): table_mean.to_csv(root_out + mean_file)
    if not(table_min.empty): table_min.to_csv(root_out + min_file)

#######################
# SPATIAL CONSISTENCY #
#######################

import pandas
import os
from pyproj import Proj
import math
from sklearn.linear_model import LinearRegression
import numpy

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])


def SpatialLinearRegression(j,table,table_list):
    val = table.loc[j,'vals']
    if val == 99999.0: return
    Length = table.shape[0]
    if j < 45: W1,W2 = 0,91
    elif 45 <= j <= Length-46: W1,W2 = j-45,j+46
    else: W1,W2 = Length-91,Length
    y_i = []
    var_i = []
    date = table.loc[j,'time']
    for n in range(len(table_list)):
        X = []
        Y = []
        row = table_list[n][table_list[n]['time'] == date]
        if row.empty: continue
        row.reset_index(drop=True, inplace=True)
        val_x0 = row.loc[0,'vals']
        if val_x0 == 99999.0: continue
        first_date = table.loc[W1,'time']
        last_date = table.loc[W2-1,'time']
        time_interval = (table_list[n]['time'] >= first_date) & (table_list[n]['time'] <= last_date)  
        subtable_n = table_list[n][time_interval]
        if subtable_n.shape[0] < 91: continue
        subtable_n.reset_index(drop=True, inplace=True)
        cont = 0
        for k in range(W1,W2):
            val_y = table.loc[k,'vals']
            if val_y == 99999.0: 
                cont += 1
                continue
            val_x = subtable_n.loc[cont,'vals']
            if val_x < 99999.0:
                X.append(val_x)
                Y.append(val_y)
            cont += 1
        X = numpy.asarray(X)
        X = numpy.reshape(X, (X.size,1))
        Y = numpy.asarray(Y)
        val_x0 = numpy.array(val_x0)
        val_x0 = numpy.reshape(val_x0, (1,1))
        if Y.size >= 5:
            reg = LinearRegression().fit(X, Y)
            Y_hat = reg.predict(X)
            y_i.append( reg.predict(val_x0) )
            var_i.append( ((Y - Y_hat) ** 2).sum() )
    y_i = numpy.asarray(y_i)
    y_i = numpy.reshape(y_i,(y_i.size,))
    var_i = numpy.asarray(var_i)
    if y_i.size >= 2:
        y_c = (1.0/((1.0/var_i).sum())) * ((y_i / var_i).sum())
        s_c = math.sqrt( y_i.size/((1.0/var_i).sum()) )
        if not(y_c - 3.5*s_c <= val <= y_c + 3.5*s_c):
            table.loc[j,'Spatial Linear Regression'] = 1
            
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

for i in range(len(L)):
    (estacion,tipo) = getEstacion(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Spatial Linear Regression',0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == estacion]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        for j in range(len(L)):
            if i != j:
                (estacion2,tipo2) = getEstacion(L[j])
                if tipo == tipo2:
                    row2 = CNE[CNE['CODIGO'] == estacion2]
                    if not(row2.empty):
                        row2.reset_index(drop=True, inplace=True)
                        lat2 = row2.loc[0,'latitud']
                        long2 = row2.loc[0,'longitud']
                        x2,y2 = p(long2, lat2)
                        H2 = row2.loc[0,'altitud']
                        if abs(H-H2) <= 100:
                            D = math.sqrt((x-x2)**2 + (y-y2)**2)
                            if D <= 200000: neighbors.append(L[j])
        if len(neighbors) >= 2:
            table_list = []
            for j in range(len(neighbors)):
                table_list.append( pandas.read_csv(root_in + neighbors[j], index_col=0) )
                table_list[j]['time'] = pandas.to_datetime(table_list[j]['time'])
            Length = table.shape[0]
            for j in range(Length):
                SpatialLinearRegression(j,table,table_list)
    table.to_csv(root_out + L[i])                     

###########################
# SPATIAL COHERENCE INDEX #
###########################
import pandas
import os
from pyproj import Proj
import math
from sklearn.linear_model import LinearRegression
import numpy
from pandas.tseries.offsets import DateOffset

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

def smoothNeighbors(j, table, table_list):
    val = table.loc[j,'vals']
    if val == 99999.0: return
    date = table.loc[j,'time']
    date_before = date - DateOffset(days=1)
    date_after = date + DateOffset(days=1)
    for n in range(len(table_list)):
        row_date = table_list[n][table_list[n]['time'] == date]
        if row_date.empty: continue
        row_date_before = table_list[n][table_list[n]['time'] == date_before]
        if row_date_before.empty: continue
        row_date_after = table_list[n][table_list[n]['time'] == date_after]
        if row_date_after.empty: continue
        index_date = row_date.index
        row_date.reset_index(drop=True, inplace=True)
        row_date_before.reset_index(drop=True, inplace=True)
        row_date_after.reset_index(drop=True, inplace=True)
        val_date = row_date.loc[0,'vals']
        val_date_before = row_date_before.loc[0,'vals']
        val_date_after = row_date_after.loc[0,'vals']
        if val_date == 99999.0 or val_date_before == 99999.0 or val_date_after == 99999.0: continue
        diffs = numpy.zeros(3)
        diffs[0] = abs(val - val_date_before)
        diffs[1] = abs(val - val_date)
        diffs[2] = abs(val - val_date_after)
        ind = numpy.argmin(diffs)
        if ind == 0: table_list[n].loc[index_date,'vals'] = val_date_before
        if ind == 2: table_list[n].loc[index_date,'vals'] = val_date_after

def correlation(X,Y):
    xu = X-X.mean()
    yu = Y-Y.mean()
    num = (xu*yu).sum()
    xu2 = xu**2
    yu2 = yu**2
    den = math.sqrt(xu2.sum())*math.sqrt(yu2.sum())
    if den>0: corr = abs(num/den)
    else: corr = 0
    return corr
    
def indexAgreement(X,Y):
    num = ( abs(Y-X) ).sum()
    yav = Y.mean()
    den = ( abs(X-yav) + abs(Y-yav) ).sum()
    if den>0: d = num/den
    else: d=0
    return d

def standardized_residuals(residuals):
    std = residuals.std()
    sqr = residuals*residuals
    sqr_sum = sqr.sum()
    length_residuals = residuals.size
    std_residuals = numpy.zeros(length_residuals)
    for r in range(length_residuals):
        std_corr = std * math.sqrt(1 - 1/length_residuals - sqr[r]/sqr_sum)
        std_residuals[r] = residuals[r] / std_corr
    return std_residuals

def spatialCoherenceIndex(j,table,table_list):
    val = table.loc[j,'vals']
    if val == 99999.0: return
    Length = table.shape[0]
    if j < 30: W1,W2 = 0,61
    elif 30 <= j <= Length-31: W1,W2 = j-30,j+31
    else: W1,W2 = Length-61,Length
    date = table.loc[j,'time']
    first_date = table.loc[W1,'time']
    last_date = table.loc[W2-1,'time']
    subtable_neighbors = []
    for n in range(len(table_list)):
        row = table_list[n][table_list[n]['time'] == date]
        if row.empty: continue
        row.reset_index(drop=True, inplace=True)
        time_interval = (table_list[n]['time'] >= first_date) & (table_list[n]['time'] <= last_date)  
        subtable_n = table_list[n][time_interval]
        if subtable_n.shape[0] < 61: continue
        subtable_n.reset_index(drop=True, inplace=True)
        subtable_neighbors.append(subtable_n)
    length_close_neighbors = len(subtable_neighbors)
    if length_close_neighbors < 2: return    
    Window = numpy.zeros((61,length_close_neighbors))
    for n in range(length_close_neighbors):
        subtable_n = subtable_neighbors[n]
        cont = 0
        for k in range(W1,W2):
            val_y = table.loc[k,'vals']
            if val_y < 99999.0: 
                val_x = subtable_n.loc[cont,'vals']
                if val_x < 99999.0: Window[cont][n] = 1
            cont += 1
    pos = 0
    for k in range(W1,W2):
        if k == j: break
        pos += 1
    if Window[pos].sum() != length_close_neighbors: return
    sum_d = 0
    Y_hat = 0
    for n in range(length_close_neighbors):
        X = []
        Y = []
        subtable_n = subtable_neighbors[n]
        cont = 0
        for k in range(W1,W2):
            if Window[cont].sum() == length_close_neighbors:
                val_y = table.loc[k,'vals']
                val_x = subtable_n.loc[cont,'vals']
                X.append(val_x)
                Y.append(val_y)
            cont += 1
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        if Y.size >= 5:
            corr = correlation(X,Y)  
            if corr >= 0.7:
                d = indexAgreement(X,Y)
                sum_d += d
                X = numpy.reshape(X, (X.size,1))
                reg = LinearRegression().fit(X, Y)
                Y0 = reg.predict(X)
                Y_hat += Y0*d 
    if sum_d > 0: Y_hat /= sum_d
    else: return
    residuals = []
    cont = 0
    cont2 = 0
    for k in range(W1,W2):
        if Window[cont].sum() == length_close_neighbors:
            val_y = table.loc[k,'vals']
            residuals.append(val_y - Y_hat[cont2])
            cont2 += 1
        cont += 1
    residuals = numpy.asarray(residuals)            
    if residuals.std() == 0.0: return
    std_residuals = standardized_residuals(residuals)    
    pos = 0
    cont = 0
    for k in range(W1,W2):
        if Window[cont].sum() == length_close_neighbors:
            if k == j: break
            pos += 1
        cont += 1
    percentile_99 = numpy.percentile(std_residuals,99)        
    if abs(std_residuals[pos]) > percentile_99: 
        table.loc[j,'Spatial Concordance Index']=1  

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

for i in range(len(L)):
    (estacion,tipo) = getEstacion(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Spatial Concordance Index',0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == estacion]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        for j in range(len(L)):
            if i != j:
                (estacion2,tipo2) = getEstacion(L[j])
                if tipo == tipo2:
                    row2 = CNE[CNE['CODIGO'] == estacion2]
                    if not(row2.empty):
                        row2.reset_index(drop=True, inplace=True)
                        lat2 = row2.loc[0,'latitud']
                        long2 = row2.loc[0,'longitud']
                        x2,y2 = p(long2, lat2)
                        H2 = row2.loc[0,'altitud']
                        if abs(H-H2) <= 100:
                            D = math.sqrt((x-x2)**2 + (y-y2)**2)
                            if D <= 200000: neighbors.append(L[j])
        if len(neighbors) >= 2:
            table_list = []
            for j in range(len(neighbors)):
                table_list.append( pandas.read_csv(root_in + neighbors[j], index_col=0) )
                table_list[j]['time'] = pandas.to_datetime(table_list[j]['time'])
            Length = table.shape[0]
            for j in range(1,Length-1):
                smoothNeighbors(j, table, table_list)
            for j in range(Length):
                spatialCoherenceIndex(j,table,table_list)  
                
    table.to_csv(root_out + L[i])            

##########################
# CORROBORACION ESPACIAL #
##########################
    
import numpy
import pandas
from pandas.tseries.offsets import DateOffset
from statsmodels import robust
import os
from pyproj import Proj
import math
from copy import deepcopy

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

def weights(x, M, MAD):
    c = 7.5
    den = c*MAD
    Lw = x.size
    u = numpy.zeros(Lw)
    for k in range(Lw):
        u[k] = x[k] - M
    u /= den 
    for k in range(Lw):
        if abs(u[k]) > 1.0: u[k] = 1.0
    return u

def mean_biweight(x, M, u):
    Lw = x.size
    num = 0
    den = 0
    for k in range(Lw):
        w = 1.0 - u[k]*u[k]
        num += (x[k] - M) * w * w
        den += w * w
    x_mean = M + num/den
    return x_mean    
     
def std_biweight(x, M, u):
    Lw = x.size
    num = 0
    den = 0
    for k in range(Lw):
        w = 1.0 - u[k]*u[k]
        w2 = 1.0 - 5*u[k]*u[k]
        num += pow(x[k] - M, 2) * pow(w, 4)
        den += w * w2
    num *= Lw
    num = math.sqrt(num)
    den = abs(den)
    x_std = num / den
    return x_std                   

def deviationSeries(table):
    table_anomaly = table.copy()
    table_anomaly['vals'] = 99999.0
    first_date = table.loc[0,'time']
    last_date = table.loc[table.shape[0]-1,'time']
    if first_date.is_leap_year: last_date_first_year = first_date + DateOffset(days=365)
    else: last_date_first_year = first_date + DateOffset(days=364)
    current_date = first_date
    while current_date <= last_date_first_year:
        W = []
        if current_date < first_date + DateOffset(days=14):
            current_date_plus_years = first_date
            while current_date_plus_years <= last_date:
                boolean = (table['time']>=current_date_plus_years) & \
                          (table['time']<=current_date_plus_years+DateOffset(days=14))
                subtable = table[boolean]
                if not(subtable.empty):
                    subtable.reset_index(drop=True, inplace=True)
                    for k in range(subtable.shape[0]):
                        value = subtable.loc[k,'vals']
                        if value < 99999.0: W.append(value)
                current_date_plus_years += DateOffset(years=1)        
        elif current_date >= first_date + DateOffset(days=14) and \
            current_date <= last_date_first_year - DateOffset(days=14):
            current_date_plus_years = current_date
            while current_date_plus_years <= last_date:
                boolean = (table['time']>=current_date_plus_years-DateOffset(7)) & \
                          (table['time']<=current_date_plus_years+DateOffset(days=7))
                subtable = table[boolean]
                if not(subtable.empty):
                    subtable.reset_index(drop=True, inplace=True)
                    for k in range(subtable.shape[0]):
                        value = subtable.loc[k,'vals']
                        if value < 99999.0: W.append(value)
                current_date_plus_years += DateOffset(years=1)
        else:
            current_date_plus_years = last_date_first_year - DateOffset(days=14)
            while current_date_plus_years <= last_date:
                boolean = (table['time']>=current_date_plus_years) & \
                          (table['time']<=current_date_plus_years+DateOffset(days=14))
                subtable = table[boolean]
                if not(subtable.empty):
                    subtable.reset_index(drop=True, inplace=True)
                    for k in range(subtable.shape[0]):
                        value = subtable.loc[k,'vals']
                        if value < 99999.0: W.append(value)
                current_date_plus_years += DateOffset(years=1)
        W = numpy.asarray(W)
        if W.size >= 5:
            ##############BIWEIGTH###########
            median = numpy.median(W)
            MAD = robust.mad(W)
            if MAD > 0:
                u = weights(W, median, MAD)
                mean = mean_biweight(W, median, u)
                std = std_biweight(W, median, u)
            else: 
                mean = median
                std = 0
            #################################
            current_date_plus_years = current_date
            while current_date_plus_years <= last_date:
                row = table[table['time'] == current_date_plus_years]
                if not(row.empty):
                    index = row.index
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals'] 
                    if value < 99999.0:
                        if std > 0:
                            Z = abs(value - mean) / std
                            if Z > 3: table_anomaly.loc[index,'vals'] = value
                current_date_plus_years += DateOffset(years=1)
        current_date += DateOffset(days=1)
    return table_anomaly    
    
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

for i in range(len(L)):
    (estacion,tipo) = getEstacion(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Spatial Corroboration',0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == estacion]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        for j in range(len(L)):
            if i != j:
                (estacion2,tipo2) = getEstacion(L[j])
                if tipo == tipo2:
                    row2 = CNE[CNE['CODIGO'] == estacion2]
                    if not(row2.empty):
                        row2.reset_index(drop=True, inplace=True)
                        lat2 = row2.loc[0,'latitud']
                        long2 = row2.loc[0,'longitud']
                        x2,y2 = p(long2, lat2)
                        H2 = row2.loc[0,'altitud']
                        if abs(H-H2) <= 100:
                            D = math.sqrt((x-x2)**2 + (y-y2)**2)
                            if D <= 200000: neighbors.append(L[j])
        if len(neighbors) >= 2:
            table_list = []
            for j in range(len(neighbors)):
                table_list.append( pandas.read_csv(root_in + neighbors[j], index_col=0) )
                table_list[j]['time'] = pandas.to_datetime(table_list[j]['time'])
            table_anomaly = deviationSeries(table)
            table_list_anomaly = []
            for j in range(len(table_list)):
                table_list_anomaly.append( deviationSeries(table_list[j]) )
            Length = table.shape[0]
            for j in range(1,Length-1):
                val = table_anomaly.loc[j,'vals']
                if val == 99999.0: continue
                date = table_anomaly.loc[j,'time']
                neighbor_anomalies = []
                for n in range(len(table_list_anomaly)):
                    date_before = date - DateOffset(days=1) 
                    row_date_before = table_list_anomaly[n][table_list_anomaly[n]['time']==date_before]
                    if not(row_date_before.empty):
                        row_date_before.reset_index(drop=True, inplace=True)
                        val_neighbor_date_before = row_date_before.loc[0,'vals']
                        if val_neighbor_date_before < 99999.0:
                            neighbor_anomalies.append(val_neighbor_date_before)
                    row_date = table_list_anomaly[n][table_list_anomaly[n]['time']==date]
                    if not(row_date.empty):
                        row_date.reset_index(drop=True, inplace=True)
                        val_neighbor_date = row_date.loc[0,'vals']
                        if val_neighbor_date < 99999.0:
                            neighbor_anomalies.append(val_neighbor_date)
                    date_after = date + DateOffset(days=1) 
                    row_date_after = table_list_anomaly[n][table_list_anomaly[n]['time']==date_after]
                    if not(row_date_after.empty):
                        row_date_after.reset_index(drop=True, inplace=True)
                        val_neighbor_date_after = row_date_after.loc[0,'vals']
                        if val_neighbor_date_after < 99999.0:
                            neighbor_anomalies.append(val_neighbor_date_after)        
                neighbor_anomalies = numpy.asarray(neighbor_anomalies)
                no_neighbors = neighbor_anomalies.size 
                if no_neighbors >= 3:
                    cont = 0
                    for k in range(no_neighbors):
                        if abs(neighbor_anomalies[k]-val) > 10: cont += 1
                    if cont == no_neighbors:
                        table.loc[j,'Spatial Corroboration'] = 1
                
    table.to_csv(root_out + L[i])  

#######################
# REPORTE TEMPERATURA #
#######################
import pandas
import os

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_reports = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/'
report = pandas.DataFrame(columns = ['codigo','time','vals','temp','Extremes',
                                     'DRI','biweigth','DCE','Persistence',
                                     'Excesive Jumps','Peaks','Consistency 1',
                                     'Consistency 2','Consistency 3','Consistency 4',
                                     'Amplitud Termica','Spatial Linear Regression',
                                     'Spatial Concordance Index','Spatial Corroboration'])

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['biweigth'] > 0)
    cond = cond | (table['DCE'] > 0)
    cond = cond | (table['Persistence'] > 0)
    cond = cond | (table['Excesive Jumps'] > 0)
    cond = cond | (table['Peaks'] > 0)
    cond = cond | (table['Consistency 1'] > 0)
    cond = cond | (table['Consistency 2'] > 0)
    cond = cond | (table['Consistency 3'] > 0)
    cond = cond | (table['Consistency 4'] > 0)
    cond = cond | (table['Amplitud Termica'] > 0)
    cond = cond | (table['Spatial Linear Regression'] > 0)
    cond = cond | (table['Spatial Concordance Index'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    if not(subtable.empty):
        report = pandas.concat([report,subtable], ignore_index=True)
        
report.drop(columns=['vals'], inplace=True) 
report.to_csv(root_reports + 'Report_Temperature1_IDEAM.csv')               

##################################
#ELIMINACION DE DATOS SOSPECHOSOS#  
##################################
import pandas
import os

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura2/'

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Error',0)
    print(str(i) + ': ' +L[i])
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['biweigth'] > 0)
    cond = cond | (table['DCE'] > 0)
    cond = cond | (table['Persistence'] > 0)
    cond = cond | (table['Excesive Jumps'] > 0)
    cond = cond | (table['Peaks'] > 0)
    cond = cond | (table['Consistency 1'] > 0)
    cond = cond | (table['Consistency 2'] > 0)
    cond = cond | (table['Consistency 3'] > 0)
    cond = cond | (table['Consistency 4'] > 0)
    cond = cond | (table['Amplitud Termica'] > 0)
    cond = cond | (table['Spatial Linear Regression'] > 0)
    cond = cond | (table['Spatial Concordance Index'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    if not(subtable.empty):
        index = subtable.index
        table.loc[index,'vals'] = 99999.0
        table.loc[index,'Error'] = 1
    
    table.drop(columns=['Extremes','DRI','biweigth','DCE','Persistence',
                        'Excesive Jumps','Peaks','Consistency 1','Consistency 2',
                        'Consistency 3','Consistency 4','Amplitud Termica',
                        'Spatial Linear Regression','Spatial Concordance Index',
                        'Spatial Corroboration'], inplace=True)
    table.to_csv(root_out + L[i])         

  
                                    #################
                                    # PRECIPITATION #
                                    #################
 
import pandas
from calendar import monthrange
import os
import numpy
import scipy.stats as stats

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'

L = os.listdir(root_in)

################################################################
# EXTREMES, DESVIACIONES RESPECTO AL RANGO INTERQUARTIL (DRI)  #
#     DESVIACION RESPECTO A LA DISTRIBUCION GAMMA (DGD)        #
################################################################

Prcpmax = 200
Prcpmin = 0
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Extremes',0)
    table.insert(table.shape[1],'DRI',0)
    table.insert(table.shape[1],'DGD',0)
    print(str(i) + ': ' +L[i])
    ##########
    #EXTREMES#
    ##########
    cond1 = table['vals'] < 99999.0
    cond2 = (table['vals'] < Prcpmin) | (table['vals'] > Prcpmax)
    cond = cond1 & cond2
    if cond.sum() > 0: table.loc[cond,['Extremes']] = 1
    ###################################################
    #DESVIACION RESPECTO AL RANGO INTERQUARTIL (DRI)  #
    #DESVIACION RESPECTO A LA DISTRIBUCION GAMMA (DGD)#
    ##################################################
    table['time'] = pandas.to_datetime(table['time'])
    current_month = 1
    last_month_first_year = 12
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    while current_month <= last_month_first_year:
        W = []
        dates = []
        current_year = first_year
        while current_year <= last_year:
            if current_month < 10: date0 = str(current_year)+'-0'+str(current_month)+'-'
            else: date0 = str(current_year)+'-'+str(current_month)+'-'
            (dummy, days_of_month) = monthrange(current_year,current_month)
            for j in range(1,days_of_month+1):
                if j < 10: date = date0+'0'+str(j)
                else: date = date0+str(j)
                date = pandas.to_datetime(date)
                row = table[table['time'] == date]
                if not(row.empty):
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals']
                    if 0.1 < value < 99999.0: 
                        W.append(value)
                        dates.append(date)
            current_year += 1
                    
        W = numpy.asarray(W)
        if W.size >= 10:
            ################DRI#################################
            ir = numpy.percentile(W,75) -  numpy.percentile(W,25) #Interquartil range
            PS = numpy.percentile(W,75) + 3 * ir
            ################DGD#################################
            alpha,loc,beta = stats.gamma.fit(W)
            Qp = stats.gamma.ppf(0.995,alpha,loc=loc,scale=beta)
            ####################################################
            for j in range(W.size):
                row = table[table['time'] == dates[j]]
                index = row.index
                value = W[j]
                if value > PS: table.loc[index,'DRI'] = 1
                if value > Qp: table.loc[index,'DGD'] = 1
        current_month += 1
    
    table.to_csv(root_out + L[i])
    
###################################
#CONTROLES DE CONTINUIDAD TEMPORAL#
###################################
import pandas
import os
import numpy
from calendar import monthrange

def get_date(current_year,current_month,day):
    if current_month < 10: date0 = str(current_year)+'-0'+str(current_month)+'-'
    else: date0 = str(current_year)+'-'+str(current_month)+'-'
    if day < 10: date = date0+'0'+str(day)
    else: date = date0+str(day)
    date = pandas.to_datetime(date)
    return date

def get_value(table,date):
    row = table[table['time'] == date]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        value = row.loc[0,'vals']
    else: value = 99999.0
    return value

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    table.insert(table.shape[1],'Persistencia',0)
    table.insert(table.shape[1],'PEDSP',0)
    #persistencia de 3 o más días de un valor constante
    j=0
    while j<table.shape[0]-2:
        if table.loc[j,'vals'] == 99999.0 or table.loc[j,'vals'] <= 0.1:
            j += 1
            continue
        W = [j]
        while table.loc[j,'vals'] == table.loc[j+1,'vals']:
            W.append(j+1)
            j += 1
            if j >= table.shape[0]-1: break
        Lw = len(W)
        if Lw >= 3:
            for k in range(Lw):
                table.loc[W[k],'Persistencia']=1
        j += 1
    #Persistencia extrema de días sin precipitación (PEDSP)   
    table['time'] = pandas.to_datetime(table['time'])
    current_month = 1
    last_month_first_year = 12
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    while current_month <= last_month_first_year:
        Lengths = []
        list_of_dates = []
        current_year = first_year
        while current_year <= last_year:
            (dummy, days_of_month) = monthrange(current_year,current_month)
            j = 1
            while j <= days_of_month:
                length = 0
                dates = []
                date = get_date(current_year,current_month,j)
                while get_value(table,date) <= 0.1 and j <= days_of_month:
                    dates.append(date)
                    length += 1
                    j += 1
                    if j<=days_of_month: date = get_date(current_year,current_month,j)
                    else: break
                if length > 0: 
                    Lengths.append(length)
                    list_of_dates.append(dates)
                j += 1
            
            current_year += 1
        Lengths = numpy.asarray(Lengths)
        if Lengths.size >= 5:
            Pth = numpy.percentile(Lengths, 99.5)
            for j in range(Lengths.size):
                if Lengths[j] > Pth:
                    dates = list_of_dates[j]
                    for k in range(len(dates)):
                        row = table[table['time'] == dates[k]]
                        if not(row.empty): table.loc[row.index,'PEDSP']=1
                        
        current_month += 1
    
    table.to_csv(root_out + L[i])
    
##########################
# CORROBORACION ESPACIAL #
##########################
    
import numpy
import pandas
from pandas.tseries.offsets import DateOffset
import os
from pyproj import Proj
import math

def getEstacion(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

def getWindow(table,date):
    month = date.month
    day = date.day
    if month == 2 and day == 29: day = 28
    year = table.loc[0,'time'].year
    current_date = pandas.to_datetime( str(year) + '-' + str(month) + '-' + str(day) )
    first_date = table.loc[0,'time']
    last_date = table.loc[table.shape[0]-1,'time']
    W = []
    while current_date <= last_date:
        if current_date - DateOffset(days=14) < first_date:
            date1 = first_date
            date2 = date1 + DateOffset(days=28)
        elif current_date + DateOffset(days=14) > last_date:
            date2 = last_date
            date1 = date2 - DateOffset(days=28)
        else:
            date1 = current_date - DateOffset(days=14)
            date2 = current_date + DateOffset(days=14)
        subtable = table[(table['time']>=date1) & (table['time']<=date2)]
        if not(subtable.empty):
            subtable.reset_index(drop=True, inplace=True)
            for k in range(subtable.shape[0]):
                val = subtable.loc[k,'vals']
                if 0.1<val<99999.0: W.append(val)
        current_date += DateOffset(years=1)
    return W

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

for i in range(len(L)):
    estacion = getEstacion(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Spatial Corroboration',0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == estacion]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        distance = []
        for j in range(len(L)):
            if i != j:
                estacion2 = getEstacion(L[j])
                row2 = CNE[CNE['CODIGO'] == estacion2]
                if not(row2.empty):
                    row2.reset_index(drop=True, inplace=True)
                    lat2 = row2.loc[0,'latitud']
                    long2 = row2.loc[0,'longitud']
                    x2,y2 = p(long2, lat2)
                    H2 = row2.loc[0,'altitud']
                    if abs(H-H2) <= 100:
                       D = math.sqrt((x-x2)**2 + (y-y2)**2)
                       if D <= 200000: 
                           neighbors.append(L[j])
                           distance.append(D)
        if len(neighbors) >= 2:
            if len(neighbors) > 7:
                distance = numpy.asarray(distance)
                indices = numpy.argsort(distance)
                neighbors2 = []
                for n in range(indices.size):
                    neighbors2.append(neighbors[indices[n]])
                neighbors = neighbors2
                neighbors = neighbors[0:7]
            table_list = []
            for j in range(len(neighbors)):
                table_list.append( pandas.read_csv(root_in + neighbors[j], index_col=0) )
                table_list[j]['time'] = pandas.to_datetime(table_list[j]['time'])
            for j in range(1,table.shape[0]-1):
                val = table.loc[j,'vals']
                if val == 99999.0: continue
                date = table.loc[j,'time']
                date_before = date - DateOffset(days=1)
                date_after = date + DateOffset(days=1)
                W = []
                for n in range(len(table_list)):
                    boolean = (table_list[n]['time']>=date_before) & (table_list[n]['time']<=date_after)
                    subtable_n = table_list[n][boolean]
                    if not(subtable_n.empty):
                        subtable_n.reset_index(drop=True, inplace=True)
                        for k in range(subtable_n.shape[0]):
                            value = subtable_n.loc[k,'vals']
                            if value < 99999.0: W.append(value)
                W = numpy.asarray(W)
                if W.size >= 5:
                    minimum = W.min()
                    maximum = W.max()
                    if minimum <= val <= maximum: continue
                    W2 = []
                    for n in range(len(table_list)):
                        W2.extend(getWindow(table_list[n],date))
                    W2 = numpy.asarray(W2)
                    if W2.size >= 20:
                        W2 = abs(W2 - val)
                        MATD = W2.min()
                        maximum = W2.max()
                        W2 = W2/maximum*100
                        MATD_percent = W2.min()
                        if MATD_percent>0: U = -45.72*math.log(MATD_percent) + 269.24
                        else: U=269.24
                        if MATD>U: table.loc[j,'Spatial Corroboration'] = 1
                            
    table.to_csv(root_out + L[i])    

#########################
# REPORTE PRECIPITACION #
#########################
import pandas
import os

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'
root_reports = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/'
report = pandas.DataFrame(columns = ['codigo','time','vals','Extremes','DRI','DGD',
                                     'Persistencia', 'PEDSP','Spatial Corroboration'])

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['DGD'] > 0)
    cond = cond | (table['Persistencia'] > 0)
    cond = cond | (table['PEDSP'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    if not(subtable.empty):
        report = pandas.concat([report,subtable], ignore_index=True)
        
report.drop(columns=['vals'], inplace=True) 
report.to_csv(root_reports + 'Report_Precipítacion1_IDEAM.csv')  

##################################
#ELIMINACION DE DATOS SOSPECHOSOS#  
##################################
import pandas
import os
import numpy

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion2/'

L = os.listdir(root_in)
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table.insert(table.shape[1],'Error',0)
    print(str(i) + ': ' +L[i])
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['DGD'] > 0)
    cond = cond | (table['Persistencia'] > 0)
    cond = cond | (table['PEDSP'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    if not(subtable.empty):
        index = subtable.index
        table.loc[index,'vals'] = 99999.0
        table.loc[index,'Error'] = 1
    
    table.drop(columns=['Extremes','DRI','DGD','Persistencia',
                        'PEDSP','Spatial Corroboration'], inplace=True)
    table.to_csv(root_out + L[i])


         