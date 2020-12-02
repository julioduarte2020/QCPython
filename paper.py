# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:11:30 2020

@author: jmduarte
"""
                            ###############
                            # TEMPERATURE #
                            ###############
# Average distance
import os
import pandas
import numpy
from pyproj import Proj
import math

def get_station_kind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

average_dist = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
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
                (station2,kind2) = get_station_kind(L[j])
                if kind == kind2:
                    row2 = CNE[CNE['CODIGO'] == station2]
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
        if len(neighbors) >= 1:
            if len(neighbors) > 5:
                distance = numpy.asarray(distance)
                indices = numpy.argsort(distance)
                distance2 = []
                for n in range(indices.size):
                    distance2.append(distance[indices[n]])
                distance = distance2
                distance = distance[0:5]
            for n in range(len(distance)):
                average_dist.append(distance[n])
    

average_dist = numpy.asarray(average_dist)
print('average distance: ' + str(average_dist.mean()))    

# Years
import os
import pandas
import numpy

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'
L = os.listdir(root_in)

years = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    days = 0.0
    for j in range(table.shape[0]):
        if table.loc[j,'vals'] < 99999.0: days += 1.0
    years.append(round(days / 365))
    print(str(i) + ': ' +L[i])

years = numpy.asarray(years)

(years < 30).sum() / years.size * 100
(years > 45).sum() / years.size * 100

# Average distance years >= 30
import os
import pandas
import numpy
from pyproj import Proj
import math

def get_station_kind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'
L = os.listdir(root_in)

years = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    days = 0.0
    for j in range(table.shape[0]):
        if table.loc[j,'vals'] < 99999.0: days += 1.0
    years.append(round(days / 365))
    print(str(i) + ': ' +L[i])

years = numpy.asarray(years)

boolean = years >= 30

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

average_dist = []
for i in range(len(L)):
    if not(boolean[i]): continue 
    (station,kind) = get_station_kind(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        distance = []
        for j in range(len(L)):
            if not(boolean[j]): continue
            if i != j:
                (station2,kind2) = get_station_kind(L[j])
                if kind == kind2:
                    row2 = CNE[CNE['CODIGO'] == station2]
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
        if len(neighbors) >= 1:
            if len(neighbors) > 5:
                distance = numpy.asarray(distance)
                indices = numpy.argsort(distance)
                distance2 = []
                for n in range(indices.size):
                    distance2.append(distance[indices[n]])
                distance = distance2
                distance = distance[0:5]
            for n in range(len(distance)):
                average_dist.append(distance[n])

average_dist = numpy.asarray(average_dist)
print('average distance: ' + str(average_dist.mean()))   

#Figure 1
import os
import pandas
import numpy
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_station_kind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/'
L = os.listdir(root_in)
CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

#MAX TEMPERATURE
lat = []
long = []
years_range = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    if kind != 'max': continue
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        if years[i] <= 5: years_range.append(1)
        elif 6 <= years[i] <= 11: years_range.append(2)
        elif 12 <= years[i] <= 18: years_range.append(3)
        elif 19 <= years[i] <= 30: years_range.append(4)
        else: years_range.append(5)

lat = numpy.asarray(lat)
long = numpy.asarray(long)
years_range = numpy.asarray(years_range)

classes=['0-5','6-11','12-18','19-30','>30']
scatter = pyplot.scatter(lat,long,years_range,years_range)
pyplot.title('Maximum Temperature')
pyplot.xlabel('Latitude')
pyplot.ylabel('Longitude')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Years range':years_range})
table.to_csv(root_out + 'Stations Maximum Temperature Range.csv')

#MEAN TEMPERATURE
lat = []
long = []
years_range = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    if kind != 'mean': continue
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        if years[i] <= 5: years_range.append(1)
        elif 6 <= years[i] <= 11: years_range.append(2)
        elif 12 <= years[i] <= 18: years_range.append(3)
        elif 19 <= years[i] <= 30: years_range.append(4)
        else: years_range.append(5)

lat = numpy.asarray(lat)
long = numpy.asarray(long)
years_range = numpy.asarray(years_range)

classes=['0-5','6-11','12-18','19-30','>30']
scatter = pyplot.scatter(lat,long,years_range,years_range)
pyplot.title('Mean Temperature')
pyplot.xlabel('Latitude')
pyplot.ylabel('Longitude')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Years range':years_range})
table.to_csv(root_out + 'Stations Mean Temperature Range.csv')

#MINIMUM TEMPERATURE
lat = []
long = []
years_range = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    if kind != 'min': continue
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        if years[i] <= 5: years_range.append(1)
        elif 6 <= years[i] <= 11: years_range.append(2)
        elif 12 <= years[i] <= 18: years_range.append(3)
        elif 19 <= years[i] <= 30: years_range.append(4)
        else: years_range.append(5)

lat = numpy.asarray(lat)
long = numpy.asarray(long)
years_range = numpy.asarray(years_range)

classes=['0-5','6-11','12-18','19-30','>30']
scatter = pyplot.scatter(lat,long,years_range,years_range)
pyplot.title('Minimum Temperature')
pyplot.xlabel('Latitude')
pyplot.ylabel('Longitude')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Years range':years_range})
table.to_csv(root_out + 'Stations Minimum Temperature Range.csv')

# Figure 2
import os
import pandas
from datetime import date
from pandas.tseries.offsets import DateOffset
import numpy
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_station_kind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

total_days = date(2019,12,31) - date(1980,1,1)
total_days = total_days.days
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura0/'
root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/'

L = os.listdir(root_in)
CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

#MAXIMUM TEMPERATURE
lat = []
long = []
missing_data_range = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    if kind != 'max': continue
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    missing = 0
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        current_date = pandas.to_datetime(date(1980,1,1))
        while current_date <= pandas.to_datetime(date(2019,12,31)):
            row2 = table[table['time'] == current_date]
            if not(row2.empty):
                row2.reset_index(drop=True, inplace=True)
                if row2.loc[0,'vals'] == 99999.0: missing += 1
            else: missing += 1
            current_date += DateOffset(days=1)
        percentage_missing = missing / total_days * 100
        if percentage_missing <= 5.0: missing_data_range.append(1)
        elif 5.0 < percentage_missing <= 10.0: missing_data_range.append(2)
        elif 10.0 < percentage_missing <= 20.0: missing_data_range.append(3)
        elif 20.0 < percentage_missing <= 30.0: missing_data_range.append(4) 
        elif 30.0 < percentage_missing <= 40.0: missing_data_range.append(5)
        elif 40.0 < percentage_missing <= 50.0: missing_data_range.append(6)
        elif 50.0 < percentage_missing <= 60.0: missing_data_range.append(7)
        else: missing_data_range.append(8)
        
lat = numpy.asarray(lat)
long = numpy.asarray(long)
missing_data_range = numpy.asarray(missing_data_range)

classes=['0-5%','5-10%','10-20%','20-30%','30-40%','40-50%','50-60%','>60%']
scatter = pyplot.scatter(lat,long,missing_data_range,missing_data_range)
pyplot.xlabel('Latitud')
pyplot.ylabel('Longitud')
pyplot.title('Maximum Temperature')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Missing data range':missing_data_range})
table.to_csv(root_out + 'Stations Maximum Temperature Missing Data Range.csv')

#MEAN TEMPERATURE
lat = []
long = []
missing_data_range = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    if kind != 'mean': continue
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    missing = 0
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        current_date = pandas.to_datetime(date(1980,1,1))
        while current_date <= pandas.to_datetime(date(2019,12,31)):
            row2 = table[table['time'] == current_date]
            if not(row2.empty):
                row2.reset_index(drop=True, inplace=True)
                if row2.loc[0,'vals'] == 99999.0: missing += 1
            else: missing += 1
            current_date += DateOffset(days=1)
        percentage_missing = missing / total_days * 100
        if percentage_missing <= 5.0: missing_data_range.append(1)
        elif 5.0 < percentage_missing <= 10.0: missing_data_range.append(2)
        elif 10.0 < percentage_missing <= 20.0: missing_data_range.append(3)
        elif 20.0 < percentage_missing <= 30.0: missing_data_range.append(4) 
        elif 30.0 < percentage_missing <= 40.0: missing_data_range.append(5)
        elif 40.0 < percentage_missing <= 50.0: missing_data_range.append(6)
        elif 50.0 < percentage_missing <= 60.0: missing_data_range.append(7)
        else: missing_data_range.append(8)
        
lat = numpy.asarray(lat)
long = numpy.asarray(long)
missing_data_range = numpy.asarray(missing_data_range)

classes=['0-5%','5-10%','10-20%','20-30%','30-40%','40-50%','50-60%','>60%']
scatter = pyplot.scatter(lat,long,missing_data_range,missing_data_range)
pyplot.xlabel('Latitud')
pyplot.ylabel('Longitud')
pyplot.title('Mean Temperature')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Missing data range':missing_data_range})
table.to_csv(root_out + 'Stations Mean Temperature Missing Data Range.csv')

#MINIMUM TEMPERATURE
lat = []
long = []
missing_data_range = []
for i in range(len(L)):
    (station,kind) = get_station_kind(L[i])
    if kind != 'min': continue
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    missing = 0
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        current_date = pandas.to_datetime(date(1980,1,1))
        while current_date <= pandas.to_datetime(date(2019,12,31)):
            row2 = table[table['time'] == current_date]
            if not(row2.empty):
                row2.reset_index(drop=True, inplace=True)
                if row2.loc[0,'vals'] == 99999.0: missing += 1
            else: missing += 1
            current_date += DateOffset(days=1)
        percentage_missing = missing / total_days * 100
        if percentage_missing <= 5.0: missing_data_range.append(1)
        elif 5.0 < percentage_missing <= 10.0: missing_data_range.append(2)
        elif 10.0 < percentage_missing <= 20.0: missing_data_range.append(3)
        elif 20.0 < percentage_missing <= 30.0: missing_data_range.append(4) 
        elif 30.0 < percentage_missing <= 40.0: missing_data_range.append(5)
        elif 40.0 < percentage_missing <= 50.0: missing_data_range.append(6)
        elif 50.0 < percentage_missing <= 60.0: missing_data_range.append(7)
        else: missing_data_range.append(8)
        
lat = numpy.asarray(lat)
long = numpy.asarray(long)
missing_data_range = numpy.asarray(missing_data_range)

classes=['0-5%','5-10%','10-20%','20-30%','30-40%','40-50%','50-60%','>60%']
scatter = pyplot.scatter(lat,long,missing_data_range,missing_data_range)
pyplot.xlabel('Latitud')
pyplot.ylabel('Longitud')
pyplot.title('Minimum Temperature')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Missing data range':missing_data_range})
table.to_csv(root_out + 'Stations Minimum Temperature Missing Data Range.csv')

#Figure 3
import os
import pandas
from pandas.tseries.offsets import DateOffset
import numpy
from matplotlib import pyplot

def getDRIWindow(table,date):
    first_date = table.loc[0,'time']
    last_date = table.loc[table.shape[0]-1,'time']
    if first_date.is_leap_year: last_date_first_year = first_date + DateOffset(days=365)
    else: last_date_first_year = first_date + DateOffset(days=364)
    if date.month == 2 and date.day > 28:
       current_date = pandas.to_datetime(str(first_date.year)+'-'+str(date.month)+'-'+str(28))     
    else: 
       current_date = pandas.to_datetime(str(first_date.year)+'-'+str(date.month)+'-'+str(date.day))
    W = []
    if current_date < first_date + DateOffset(days=2):
        current_date_plus_years = first_date
        while current_date_plus_years <= last_date:
            for k in range(5):
                row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                if not(row.empty):
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals']
                    W.append(value)
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
                    W.append(value)
            current_date_plus_years += DateOffset(years=1)
    else:
        current_date_plus_years = last_date_first_year - DateOffset(days=4)
        while current_date_plus_years <= last_date:
            for k in range(5):
                row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                if not(row.empty):
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals']
                    W.append(value)
            current_date_plus_years += DateOffset(years=1)
    W = numpy.asarray(W)
    return W

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
L = os.listdir(root_in)
table = pandas.read_csv(root_in + L[0], index_col=0)
subtable = table[table['vals']<99999.0].copy()
subtable['time'] = pandas.to_datetime(subtable['time'])
subtable.reset_index(drop=True, inplace=True)
values = subtable['vals'].to_numpy()
upper_limit = []
lower_limit = []
for i in range(subtable.shape[0]):
    date = subtable.loc[i,'time']
    W = getDRIWindow(subtable,date)
    if (W.size >= 3):
       median = numpy.median(W)
       ir = numpy.percentile(W,75) -  numpy.percentile(W,25) #Interquartil range
       ssd = ir / 1.349 #pseudo standard deviation
       upper_limit.append(median + 3.0*ssd)
       lower_limit.append(median - 3.0*ssd)
    else:
        val = subtable.loc[i,'vals']
        upper_limit.append(val)
        lower_limit.append(val)
upper_limit = numpy.asarray(upper_limit)
lower_limit = numpy.asarray(lower_limit)       

dates = subtable['time'].to_numpy()

pyplot.plot(dates, upper_limit, 'r.', label='upper limit')
pyplot.plot(dates, lower_limit, 'r.', label='lower limit')
pyplot.plot(dates, values, 'b.', label='data')
pyplot.xlabel('Dates')
pyplot.ylabel('Temperature (°C)')
pyplot.title('Deviation with respect to the Interquartil Range')
pyplot.legend()
pyplot.show()

#Percentage of atypical values
import os
import pandas

def get_station_kind(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

min_temp_atypical = 0
mean_temp_atypical = 0
max_temp_atypical = 0
min_temp_total = 0
mean_temp_total = 0
max_temp_total = 0

root = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura2/'
L = os.listdir(root)

for i in range(len(L)):
    file = L[i]
    (station,kind) = get_station_kind(file)
    print(L[i])
    table = pandas.read_csv(root + L[i], index_col=0)
    length = table.shape[0]
    if kind == 'min': min_temp_total += length
    elif kind == 'mean': mean_temp_total += length
    else: max_temp_total += length
    subtable = table[table['Error']==1]
    length2 = subtable.shape[0] 
    if kind == 'min': min_temp_atypical += length2
    elif kind == 'mean': mean_temp_atypical += length2
    else: max_temp_atypical += length2

print(min_temp_atypical/min_temp_total)
print(mean_temp_atypical/mean_temp_total)
print(max_temp_atypical/max_temp_total)
       
#Figure 5
#Filling of missing data
import os
import pandas
from matplotlib import pyplot

root_1 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura2/'
root_2 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura3/'

L1 = os.listdir(root_1)
L2 = os.listdir(root_2)

table1 = pandas.read_csv(root_1 + L1[0], index_col=0)
table2 = pandas.read_csv(root_2 + L2[0], index_col=0)

table1['time'] = pandas.to_datetime(table1['time'])
subtable1 = table1[table1['vals']<99999.0]
subtable1.reset_index(drop=True, inplace=True)
values1 = subtable1['vals'].to_numpy()
dates1 = subtable1['time'].to_numpy()

table2['time'] = pandas.to_datetime(table2['time'])
subtable2 = table2[table2['vals']<99999.0]
subtable2.reset_index(drop=True, inplace=True)
values2 = subtable2['vals'].to_numpy()
dates2 = subtable2['time'].to_numpy()

pyplot.plot(dates2,values2,'r.',label='Data Filled')
pyplot.plot(dates1,values1,'b.',label='Data')
pyplot.xlabel('Dates')
pyplot.ylabel('Temperature (°C)')
pyplot.title('Filling of Missing Data')
pyplot.legend()
pyplot.show()

#Percentage of data filled using Kriging
import os
import pandas

total_missing = 0
missing_filled = 0

root_1 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura2/'
root_2 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura3/'

L = os.listdir(root_1)
for i in range(len(L)):
    print(L[i])
    table1 = pandas.read_csv(root_1 + L[i], index_col=0)
    table2 = pandas.read_csv(root_2 + L[i], index_col=0)
    subtable1 = table1[table1['vals']==99999.0]
    subtable2 = table2[table2['vals']==99999.0]
    total_missing += subtable1.shape[0]
    missing_filled += subtable1.shape[0] - subtable2.shape[0]

print(missing_filled/total_missing)
    
#QUALITY INDEX
import os
import pandas
import datetime
import numpy

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2019,12,31)
total_days = (end-start).days 
L = os.listdir(root_in)

#Percentage of days in the 1980 to 2019 range
P = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    boolean = (table['time']>=start) & (table['time']<=end)
    subtable = table[boolean]
    if subtable.empty:
        P.append(0)
        continue
    days = 0.0
    subtable.reset_index(drop=True, inplace=True)
    for j in range(subtable.shape[0]):
        if subtable.loc[j,'vals'] < 99999.0: days += 1.0
    P.append(days/total_days*100)
    print(str(i) + ': ' +L[i])

P = numpy.asarray(P)
P = P.mean()
print(P)

#Percentage of gaps
Qgaps = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    ngap = 0
    max_length = 0
    length = 0
    n = table.shape[0]
    for j in range(n):
        if table.loc[j,'vals'] == 99999.0: 
            ngap += 1.0
            length += 1
        else: length = 0
        if length > max_length: max_length = length
    Qgaps.append(100 - 100*((2*ngap+max_length)/n))
    print(str(i) + ': ' +L[i])
    
Qgaps = numpy.asarray(Qgaps)
Qgaps = Qgaps.mean()
print(Qgaps)

#Percentage of outliers
Qoutliers = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    n = table.shape[0]
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
    if subtable.empty:
       Qoutliers.append(100)
       continue
    nout = subtable.shape[0]
    Qoutliers.append(100 - 100*(nout/n))
    print(str(i) + ': ' +L[i])
    
Qoutliers = numpy.asarray(Qoutliers)
Qoutliers = Qoutliers.mean()
print(Qoutliers)

Q = (P+Qgaps+Qoutliers)/3

#QUALITY INDEX 2, Figure 4a
import os
import pandas
import datetime
import numpy
from matplotlib import pyplot

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2019,12,31)
total_days = (end-start).days 
L = os.listdir(root_in)

Qrange=numpy.arange(40,101,5)
nstations = numpy.zeros(Qrange.size)

for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    #Percentage of days in the 1980 to 2019 range
    boolean = (table['time']>=start) & (table['time']<=end)
    subtable = table[boolean]
    if subtable.empty: P = 0
    else:
        days = 0.0
        subtable.reset_index(drop=True, inplace=True)
        for j in range(subtable.shape[0]):
            if subtable.loc[j,'vals'] < 99999.0: days += 1.0
        P = days/total_days*100
    #Percentage of gaps
    ngap = 0
    max_length = 0
    length = 0
    n = table.shape[0]
    for j in range(n):
        if table.loc[j,'vals'] == 99999.0: 
            ngap += 1.0
            length += 1
        else: length = 0
        if length > max_length: max_length = length
    Qgaps = 100 - 100*((2*ngap+max_length)/n)
    #Percentage of outliers
    n = table.shape[0]
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
    if subtable.empty: Qoutliers=100
    else:
        nout = subtable.shape[0]
        Qoutliers = 100 - 100*(nout/n)
    Q = (P+Qgaps+Qoutliers)/3 
    for j in range(Qrange.size):
        if Qrange[j]-2.5 <= Q < Qrange[j]+2.5:
            nstations[j] += 1
            continue
    print(str(i) + ': ' +L[i])

nstations = (nstations / len(L))*100
    
pyplot.bar(Qrange, nstations)
pyplot.xlabel('Quality Index (Q)')
pyplot.ylabel('Percentage of stations (%)')
pyplot.title('Temperature')
pyplot.show()

#Figure 5a
import os
import pandas
import datetime
import numpy
from matplotlib import pyplot

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2019,12,31)
total_days = (end-start).days 
L = os.listdir(root_in)

dates = pandas.date_range(start,end)
nstations1 = numpy.zeros(dates.size)
nstations2 = numpy.zeros(dates.size)
nstations3 = numpy.zeros(dates.size)

for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    #Percentage of days in the 1980 to 2019 range
    boolean = (table['time']>=start) & (table['time']<=end)
    subtable0 = table[boolean]
    if subtable0.empty: P = 0
    else:
        days = 0.0
        subtable0.reset_index(drop=True, inplace=True)
        for j in range(subtable0.shape[0]):
            if subtable0.loc[j,'vals'] < 99999.0: days += 1.0
        P = days/total_days*100
    #Percentage of gaps
    ngap = 0
    max_length = 0
    length = 0
    n = table.shape[0]
    for j in range(n):
        if table.loc[j,'vals'] == 99999.0: 
            ngap += 1.0
            length += 1
        else: length = 0
        if length > max_length: max_length = length
    Qgaps = 100 - 100*((2*ngap+max_length)/n)
    #Percentage of outliers
    n = table.shape[0]
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
    subtable1 = table[cond]
    if subtable1.empty: Qoutliers=100
    else:
        nout = subtable1.shape[0]
        Qoutliers = 100 - 100*(nout/n)
    Q = (P+Qgaps+Qoutliers)/3 
    #Selected Q, P, Qoutliers
    good_quality = 1
    if Q >= 80 and P >= 80 and Qoutliers >= 94: good_quality = 3
    elif Q >= 70 and P >= 70 and Qoutliers >= 88: good_quality = 2 
    if not(subtable0.empty):
        for j in range(subtable0.shape[0]):
            if subtable0.loc[j,'vals'] < 99999.0:
                date = subtable0.loc[j,'time']
                pos = dates.get_loc(date)
                if good_quality == 3: nstations1[pos] += 1
                elif good_quality == 2: nstations2[pos] += 1
                else: nstations3[pos] += 1

pyplot.plot(dates, nstations1, 'r', label='Q>=80%, completeness>=80%, outliers<=6%')
pyplot.plot(dates, nstations2, 'b', label='Q>=70%, completeness>=70%, outliers<=12%')
pyplot.plot(dates, nstations3, 'g', label='Q<70% or completeness<70% or outliers>12%')
pyplot.xlabel('Dates')
pyplot.ylabel('Stations')
pyplot.title('Temperature')
pyplot.legend()
pyplot.show() 
       
#Probability false negatives
import pandas
from pandas.tseries.offsets import DateOffset
import os
import numpy
from statsmodels import robust
from pyproj import Proj
import math
from sklearn.linear_model import LinearRegression
from patsy import dmatrix
import statsmodels.api as sm
from random import seed
from random import random
from random import randint
seed(1)

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

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return (tokenize[0],tokenize[1])

def getWindowDRIbiweight(table, date):
    first_date = table.loc[0,'time']
    first_year = first_date.year
    last_date = table.loc[table.shape[0]-1,'time']
    if first_date.is_leap_year: last_date_first_year = first_date + DateOffset(days=365)
    else: last_date_first_year = first_date + DateOffset(days=364)
    if date.month == 2 and date.day>28:
        current_date = pandas.to_datetime(str(first_year)+'-'+str(2)+'-'+str(28))
    else:    
        current_date = pandas.to_datetime(str(first_year)+'-'+str(date.month)+'-'+str(date.day))
    W = []
    if current_date < first_date + DateOffset(days=2):
        current_date_plus_years = first_date
        while current_date_plus_years <= last_date:
            for k in range(5):
                row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                if not(row.empty):
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals']
                    W.append(value)
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
                    W.append(value)
            current_date_plus_years += DateOffset(years=1)
    else:
        current_date_plus_years = last_date_first_year - DateOffset(days=4)
        while current_date_plus_years <= last_date:
            for k in range(5):
                row = table[table['time'] == current_date_plus_years + DateOffset(days=k)]
                if not(row.empty):
                    row.reset_index(drop=True, inplace=True)
                    value = row.loc[0,'vals']
                    W.append(value)
            current_date_plus_years += DateOffset(years=1)
    W = numpy.asarray(W)
    return W    

def desviacionCicloEstacional(table,date):
    current_year = date.year
    subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
    if subtable.shape[0] < 100: return False
    x = pandas.DatetimeIndex(subtable['time']).dayofyear
    x = numpy.asarray(x)
    day = date.dayofyear
    subindex = numpy.where(x==day)[0][0]
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
        desv_date = desviacion[subindex]
        if desv_date < desv_01 or desv_date > desv_99: return True
        else: return False
    except: return False
    
def searchList(list, item):
    for i in range(len(list)):
        if list[i] == item:
            return (True,i)
    return (False,-1)

def SpatialLinearRegression(j,table,table_list):
    val = table.loc[j,'vals']
    if val == 99999.0: return False
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
        if y_c - 3.5*s_c <= val <= y_c + 3.5*s_c: return False
        else: return True

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
    if val == 99999.0: return False
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
    if length_close_neighbors < 2: return False    
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
    if Window[pos].sum() != length_close_neighbors: return False
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
    else: return False
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
    if residuals.std() > 0.0: std_residuals = standardized_residuals(residuals)
    else: return False    
    pos = 0
    cont = 0
    for k in range(W1,W2):
        if Window[cont].sum() == length_close_neighbors:
            if k == j: break
            pos += 1
        cont += 1
    percentile_99 = numpy.percentile(std_residuals,99)        
    if abs(std_residuals[pos]) > percentile_99: return True  

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

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura2/'
L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')
CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str})
selected_files = []
for i in range(100):
    selected_files.append(L[randint(0,len(L)-1)])

Tmax = 50
Tmin = -10
probability_false_negatives = []
for i in range(len(selected_files)):
    table = pandas.read_csv(root_in + selected_files[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    (station,kind) = get_station(selected_files[i])
    print(str(i) + ': ' +selected_files[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        for j in range(len(L)):
            if i != j:
                (station2,kind2) = get_station(L[j])
                if kind == kind2:
                    row2 = CNE[CNE['CODIGO'] == station2]
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
    subtable = table[table['vals'] < 99999.0].copy()
    subtable.reset_index(drop=True, inplace=True)
    subtable.insert(subtable.shape[1],'Suspicious',0)
    length = subtable.shape[0]
    index_list = []
    for j in range(200):
        index = randint(0,length-1)
        index_list.append(index)
        val = subtable.loc[index,'vals']
        coin = random()
        if coin < 0.5:  multiplier = 0.5 + random()*0.25
        else:  multiplier = 1.25 + random()*0.25
        new_val = multiplier*val
        subtable.loc[index,'vals'] = new_val
        if new_val > Tmax or new_val < Tmin:
            subtable.loc[index,'Suspicious'] = 0.0667
    for j in range(200):
        index = index_list[j]
        date = subtable.loc[index,'time']
        W = getWindowDRIbiweight(subtable, date)
        if (W.size >= 3):
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
            value = subtable.loc[index,'vals'] 
            if ssd > 0:
               Z = abs(value - median) / ssd
               if Z > 3: subtable.loc[index,'Suspicious'] += 0.0667*0.5
            if std > 0:
               Z = abs(value - mean) / std
               if Z > 3: subtable.loc[index,'Suspicious'] += 0.0667
        #########Desviation respect to the stational cycle (DCE)#######
        if desviacionCicloEstacional(subtable,date):
            subtable.loc[index,'Suspicious'] += 0.0667
    ##############Peaks and short duration peaks#############
    diff = []
    for j in range(1,subtable.shape[0]):
        current_date = subtable.loc[j,'time']
        previous_date = current_date - DateOffset(days=1)
        if subtable.loc[j-1,'time'] == previous_date:
            diff.append( abs(subtable.loc[j,'vals'] - subtable.loc[j-1,'vals']) )
    diff = numpy.asarray(diff)
    Perc = numpy.percentile(diff,99)
    for j in range(200):
        index = index_list[j]
        current_date = subtable.loc[index,'time']
        previous_date = current_date - DateOffset(days=1)
        if index>0 and index<length-1:
            if subtable.loc[index-1,'time'] == previous_date:
                current_diff = abs(subtable.loc[index,'vals'] - subtable.loc[index-1,'vals'])
                if current_diff > Perc: subtable.loc[index,'Suspicious'] += 0.0667*0.166
                next_date = current_date + DateOffset(days=1)
                if subtable.loc[index+1,'time'] == next_date:
                    next_diff = abs(subtable.loc[index,'vals'] - subtable.loc[index+1,'vals'])
                    if current_diff > Perc and next_diff > Perc:
                        subtable.loc[index,'Suspicious'] += 0.0667*0.166
    ################Consistency among temperatures###############
    max_file = station + '_max_temperature.csv'
    mean_file = station + '_mean_temperature.csv'
    min_file = station + '_min_temperature.csv'
    table_max = pandas.DataFrame()
    table_mean = pandas.DataFrame()
    table_min = pandas.DataFrame() 
    if kind == 'max':
        table_max = subtable
        (found,pos)=searchList(L, mean_file)
        if found: 
            table_mean = pandas.read_csv(root_in + L[pos], index_col=0)
            table_mean['time'] = pandas.to_datetime(table_mean['time'])
        (found,pos)=searchList(L, min_file)
        if found: 
            table_min = pandas.read_csv(root_in + L[pos], index_col=0)
            table_min['time'] = pandas.to_datetime(table_min['time'])
    elif kind == 'min':
        table_min = subtable
        (found,pos)=searchList(L, mean_file)
        if found: 
            table_mean = pandas.read_csv(root_in + L[pos], index_col=0)
            table_mean['time'] = pandas.to_datetime(table_mean['time'])
        (found,pos)=searchList(L, max_file)
        if found: 
            table_max = pandas.read_csv(root_in + L[pos], index_col=0)
            table_max['time'] = pandas.to_datetime(table_max['time'])
    else:
        table_mean = subtable
        (found,pos)=searchList(L, max_file)
        if found: 
            table_max = pandas.read_csv(root_in + L[pos], index_col=0)
            table_max['time'] = pandas.to_datetime(table_max['time'])
        (found,pos)=searchList(L, min_file)
        if found: 
            table_min = pandas.read_csv(root_in + L[pos], index_col=0)
            table_min['time'] = pandas.to_datetime(table_min['time'])
    if not(table_max.empty) and not(table_mean.empty) and not(table_min.empty):
        # Consistency 2
        diff = []
        for j in range(subtable.shape[0]):
            date = subtable.loc[j,'time']        
            row_max = table_max[table_max['time'] == date]
            if not(row_max.empty):
               row_max.reset_index(drop=True, inplace=True)
               T_max = row_max.loc[0,'vals']
               if T_max == 99999.0: continue
            else: continue
            row_mean = table_mean[table_mean['time'] == date]
            if not(row_mean.empty):
               row_mean.reset_index(drop=True, inplace=True)
               T_mean = row_mean.loc[0,'vals']
               if T_mean == 99999.0: continue
            else: continue
            row_min = table_min[table_min['time'] == date]
            if not(row_min.empty):
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
            else: continue
            T_av = 0.5 * (T_min + T_max)
            diff.append(abs(T_mean - T_av))
        diff = numpy.asarray(diff)
        Thr = numpy.percentile(diff, 99.9)
        for j in range(200):
            index = index_list[j]
            date = subtable.loc[index,'time']        
            row_max = table_max[table_max['time'] == date]
            if not(row_max.empty):
               row_max.reset_index(drop=True, inplace=True)
               T_max = row_max.loc[0,'vals']
               if T_max == 99999.0: continue
            else: continue
            row_mean = table_mean[table_mean['time'] == date]
            if not(row_mean.empty):
               row_mean.reset_index(drop=True, inplace=True)
               T_mean = row_mean.loc[0,'vals']
               if T_mean == 99999.0: continue
            else: continue
            row_min = table_min[table_min['time'] == date]
            if not(row_min.empty):
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
            else: continue
            # Consistency 1
            if not( T_min < T_mean < T_max ): subtable.loc[index,'Suspicious'] += 0.0667
            T_av = 0.5 * (T_min + T_max)
            # Consistency 2
            if abs(T_mean - T_av) > Thr: subtable.loc[index,'Suspicious'] += 0.0667*0.5
    #Special cases
    if table_max.empty and not(table_mean.empty) and not(table_min.empty):
        for j in range(200):
            index = index_list[j]
            date = subtable.loc[index,'time']        
            row_mean = table_mean[table_mean['time'] == date]
            if not(row_mean.empty):
               row_mean.reset_index(drop=True, inplace=True)
               T_mean = row_mean.loc[0,'vals']
               if T_mean == 99999.0: continue
            else: continue
            row_min = table_min[table_min['time'] == date]
            if not(row_min.empty):
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
            else: continue
            # Consistency 1
            if not( T_min < T_mean ): subtable.loc[index,'Suspicious'] += 0.0667
    if not(table_max.empty) and not(table_mean.empty) and table_min.empty:
        for j in range(200):
            index = index_list[j]
            date = subtable.loc[index,'time']        
            row_max = table_max[table_max['time'] == date]
            if not(row_max.empty):
               row_max.reset_index(drop=True, inplace=True)
               T_max = row_max.loc[0,'vals']
               if T_max == 99999.0: continue
            else: continue
            row_mean = table_mean[table_mean['time'] == date]
            if not(row_mean.empty):
               row_mean.reset_index(drop=True, inplace=True)
               T_mean = row_mean.loc[0,'vals']
               if T_mean == 99999.0: continue
            else: continue
            # Consistency 1
            if not( T_mean < T_max ): subtable.loc[index,'Suspicious'] += 0.0667
    # Consistencies 3 and 4
    if not(table_max.empty) and not(table_min.empty):
        for j in range(200):
            index = index_list[j]
            date = subtable.loc[index,'time']        
            row_max = table_max[table_max['time'] == date]
            if not(row_max.empty):
               row_max.reset_index(drop=True, inplace=True)
               T_max = row_max.loc[0,'vals']
               if T_max == 99999.0: continue
            else: continue
            row_min = table_min[table_min['time'] == date]
            if not(row_min.empty):
               row_min.reset_index(drop=True, inplace=True)
               T_min = row_min.loc[0,'vals']
               if T_min == 99999.0: continue
            else: continue    
            date_before = date - DateOffset(days=1)
            date_after = date + DateOffset(days=1)
            row_min_before = table_min[table_min['time'] == date_before]
            row_min_after = table_min[table_min['time'] == date_after]
            if not(row_min_before.empty) and not(row_min_after.empty):
               row_min_before.reset_index(drop=True, inplace=True)
               T_min_before = row_min_before.loc[0,'vals']
               row_min_after.reset_index(drop=True, inplace=True)
               T_min_after = row_min_after.loc[0,'vals']
               #Consistency 3
               if T_min_before < 99999.0 and T_min_after < 99999.0:
                   if not(T_min_before <= T_max >= T_min_after):
                       subtable.loc[index,'Suspicious'] += 0.0667*0.166
            row_max_before = table_max[table_max['time'] == date_before]
            row_max_after = table_max[table_max['time'] == date_after]
            if not(row_max_before.empty) and not(row_max_after.empty):
               row_max_before.reset_index(drop=True, inplace=True)
               T_max_before = row_max_before.loc[0,'vals']
               row_max_after.reset_index(drop=True, inplace=True)
               T_max_after = row_max_after.loc[0,'vals']
               #Consistency 4
               if T_max_before < 99999.0 and T_max_after < 99999.0:
                   if not(T_max_before >= T_min <= T_max_after):
                       subtable.loc[index,'Suspicious'] += 0.0667*0.166
            #Amplitud Térmica
            Amplitud_termica = T_max - T_min
            if not( 0.01 <= Amplitud_termica <= 30 ):
                subtable.loc[index,'Suspicious'] += 0.0667*0.166
    if len(neighbors) >= 2:
        #########Spatial Consistency#########            
        for j in range(200):
            index = index_list[j]
            if SpatialLinearRegression(index,subtable,table_list):
                subtable.loc[index,'Suspicious'] += 0.0667
        #########Spatial Coherence Index #######
        for j in range(200):
            index = index_list[j]
            smoothNeighbors(index, subtable, table_list)            
        for j in range(200):
            index = index_list[j]
            if SpatialLinearRegression(index,subtable,table_list):
                subtable.loc[index,'Suspicious'] += 0.0667       
        #########Spatial Corroboration #######        
        table_anomaly = deviationSeries(subtable)
        table_list_anomaly = []
        for j in range(len(table_list)):
            table_list_anomaly.append( deviationSeries(table_list[j]) )
        for j in range(200):
            index = index_list[j]
            val = table_anomaly.loc[index,'vals']
            if val == 99999.0: continue
            date = table_anomaly.loc[index,'time']
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
                    subtable.loc[index,'Suspicious'] += 0.0667

    table_suspicious = subtable[subtable['Suspicious']>0]
    
    prob_false_negatives = (200-table_suspicious.shape[0])/200   
    print(prob_false_negatives)
    probability_false_negatives.append(prob_false_negatives)
    
probability_false_negatives = numpy.asarray(probability_false_negatives)

print('probability false negatives: '+ str(probability_false_negatives.mean())) 

#probability false positives
import os
import pandas

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Temperatura1/'
L = os.listdir(root_in)

cont = 0.0
cont2 = 0.0

for i in range(len(L)):
    print(str(i) + ': ' +L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    cont += table.shape[0]
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
    cont2 += subtable.shape[0]        

probability_false_positives = cont2/cont
print('probability false positives: '+ str(probability_false_positives)) 

       
                                #################
                                # PRECIPITATION #
                                #################

import os
import pandas
import numpy
from pyproj import Proj
import math

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

average_dist = []
for i in range(len(L)):
    station = get_station(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
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
                station2 = get_station(L[j])
                row2 = CNE[CNE['CODIGO'] == station2]
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
        if len(neighbors) >= 1:
            if len(neighbors) > 5:
                distance = numpy.asarray(distance)
                indices = numpy.argsort(distance)
                distance2 = []
                for n in range(indices.size):
                    distance2.append(distance[indices[n]])
                distance = distance2
                distance = distance[0:5]
            for n in range(len(distance)):
                average_dist.append(distance[n])
    
average_dist = numpy.asarray(average_dist)
print('average distance: ' + str(average_dist.mean()))  

# Years
import os
import pandas
import numpy

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'
L = os.listdir(root_in)

years = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    days = 0.0
    for j in range(table.shape[0]):
        if table.loc[j,'vals'] < 99999.0: days += 1.0
    years.append(round(days / 365))
    print(str(i) + ': ' +L[i])

years = numpy.asarray(years)

(years < 30).sum() / years.size * 100
(years > 45).sum() / years.size * 100

# Average distance years >= 30
import os
import pandas
import numpy
from pyproj import Proj
import math

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'
L = os.listdir(root_in)

years = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    days = 0.0
    for j in range(table.shape[0]):
        if table.loc[j,'vals'] < 99999.0: days += 1.0
    years.append(round(days / 365))
    print(str(i) + ': ' +L[i])

years = numpy.asarray(years)

boolean = years >= 30

CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 

L = os.listdir(root_in)
p = Proj(proj='utm',zone=18,ellps='WGS84')

average_dist = []
for i in range(len(L)):
    if not(boolean[i]): continue
    station = get_station(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'latitud']
        long = row.loc[0,'longitud']
        x,y = p(long, lat)
        H = row.loc[0,'altitud']
        neighbors = []
        distance = []
        for j in range(len(L)):
            if not(boolean[j]): continue
            if i != j:
                station2 = get_station(L[j])
                row2 = CNE[CNE['CODIGO'] == station2]
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
        if len(neighbors) >= 1:
            if len(neighbors) > 5:
                distance = numpy.asarray(distance)
                indices = numpy.argsort(distance)
                distance2 = []
                for n in range(indices.size):
                    distance2.append(distance[indices[n]])
                distance = distance2
                distance = distance[0:5]
            for n in range(len(distance)):
                average_dist.append(distance[n])
    

average_dist = numpy.asarray(average_dist)
print('average distance: ' + str(average_dist.mean()))  
 
#Figure 1
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'
L = os.listdir(root_in)
CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 
lat = []
long = []
years_range = []
for i in range(len(L)):
    station = get_station(L[i])
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        if years[i] <= 5: years_range.append(1)
        elif 6 <= years[i] <= 11: years_range.append(2)
        elif 12 <= years[i] <= 18: years_range.append(3)
        elif 19 <= years[i] <= 30: years_range.append(4)
        else: years_range.append(5)

lat = numpy.asarray(lat)
long = numpy.asarray(long)
years_range = numpy.asarray(years_range)

classes=['0-5','6-11','12-18','19-30','>30']
scatter = pyplot.scatter(lat,long,years_range,years_range)
pyplot.xlabel('Latitud')
pyplot.ylabel('Longitud')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/'
table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Years range':years_range})
table.to_csv(root_out + 'Stations Precipitation Range.csv')

#Figure 2        
import os
import pandas
from datetime import date
from pandas.tseries.offsets import DateOffset
import numpy
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

total_days = date(2019,12,31) - date(1980,1,1)
total_days = total_days.days
root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion0/'

L = os.listdir(root_in)
CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str}) 
lat = []
long = []
missing_data_range = []
for i in range(len(L)):
    station = get_station(L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    missing = 0
    print(str(i) + ': ' +L[i])
    row = CNE[CNE['CODIGO'] == station]
    if not(row.empty):
        row.reset_index(drop=True, inplace=True)
        lat.append(row.loc[0,'latitud'])
        long.append(row.loc[0,'longitud'])
        current_date = pandas.to_datetime(date(1980,1,1))
        while current_date <= pandas.to_datetime(date(2019,12,31)):
            row2 = table[table['time'] == current_date]
            if not(row2.empty):
                row2.reset_index(drop=True, inplace=True)
                if row2.loc[0,'vals'] == 99999.0: missing += 1
            else: missing += 1
            current_date += DateOffset(days=1)
        percentage_missing = missing / total_days * 100
        if percentage_missing <= 5.0: missing_data_range.append(1)
        elif 5.0 < percentage_missing <= 10.0: missing_data_range.append(2)
        elif 10.0 < percentage_missing <= 20.0: missing_data_range.append(3)
        elif 20.0 < percentage_missing <= 30.0: missing_data_range.append(4) 
        elif 30.0 < percentage_missing <= 40.0: missing_data_range.append(5)
        elif 40.0 < percentage_missing <= 50.0: missing_data_range.append(6)
        elif 50.0 < percentage_missing <= 60.0: missing_data_range.append(7)
        else: missing_data_range.append(8)
        
lat = numpy.asarray(lat)
long = numpy.asarray(long)
missing_data_range = numpy.asarray(missing_data_range)

classes=['0-5%','5-10%','10-20%','20-30%','30-40%','40-50%','50-60%','>60%']
scatter = pyplot.scatter(lat,long,missing_data_range,missing_data_range)
pyplot.xlabel('Latitud')
pyplot.ylabel('Longitud')
pyplot.legend(handles=scatter.legend_elements()[0], labels=classes)
pyplot.show()

root_out = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/'
table = pandas.DataFrame({'Latitud':lat, 'Longitud':long, 'Missing data range':missing_data_range})
table.to_csv(root_out + 'Stations Precipitation Missing Data Range.csv')

#Figure 4
import os
import pandas
from calendar import monthrange
import numpy
from matplotlib import pyplot

def getWindowDRIGamma(table,date):
    current_month = date.month
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    W = []
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
                W.append(value)
        current_year += 1
                
    W = numpy.asarray(W)
    return W

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'
L = os.listdir(root_in)
table = pandas.read_csv(root_in + L[0], index_col=0)
subtable = table[table['vals']<99999.0].copy()
subtable['time'] = pandas.to_datetime(subtable['time'])
subtable.reset_index(drop=True, inplace=True)
values = subtable['vals'].to_numpy()
upper_limit = []
for i in range(subtable.shape[0]):
    date = subtable.loc[i,'time']
    W = getWindowDRIGamma(subtable, date)
    if (W.size >= 10):
       median = numpy.median(W)
       ir = numpy.percentile(W,75) -  numpy.percentile(W,25) #Interquartil range
       PS = numpy.percentile(W,75) + 3 * ir
       upper_limit.append(PS)
    else:
        val = subtable.loc[i,'vals']
        upper_limit.append(val)
upper_limit = numpy.asarray(upper_limit)

dates = subtable['time'].to_numpy()

pyplot.plot(dates, upper_limit, 'r.', label='upper limit')
pyplot.plot(dates, values, 'b.', label='data')
pyplot.xlabel('Dates')
pyplot.ylabel('Precipitation (mm)')
pyplot.title('Deviation with respect to the Interquartil Range')
pyplot.legend()
pyplot.show()

#Percentage of atypical values
import os
import pandas

atypical = 0
total = 0

root = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion2/'
L = os.listdir(root)

for i in range(len(L)):
    print(L[i])
    table = pandas.read_csv(root + L[i], index_col=0)
    length = table.shape[0]
    total += length
    subtable = table[table['Error']==1]
    length2 = subtable.shape[0] 
    atypical += length2
    
print(atypical/total)

#Figure 6
#Filling of missing data
import os
import pandas
from matplotlib import pyplot

root_1 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion2/'
root_2 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion3/'

L1 = os.listdir(root_1)
L2 = os.listdir(root_2)

table1 = pandas.read_csv(root_1 + L1[0], index_col=0)
table2 = pandas.read_csv(root_2 + L2[0], index_col=0)

table1['time'] = pandas.to_datetime(table1['time'])
subtable1 = table1[table1['vals']<99999.0]
subtable1.reset_index(drop=True, inplace=True)
values1 = subtable1['vals'].to_numpy()
dates1 = subtable1['time'].to_numpy()

table2['time'] = pandas.to_datetime(table2['time'])
subtable2 = table2[table2['vals']<99999.0]
subtable2.reset_index(drop=True, inplace=True)
values2 = subtable2['vals'].to_numpy()
dates2 = subtable2['time'].to_numpy()

pyplot.plot(dates2,values2,'r.',label='Data Filled')
pyplot.plot(dates1,values1,'b.',label='Data')
pyplot.xlabel('Dates')
pyplot.ylabel('Precipitation (mm)')
pyplot.title('Filling of Missing Data')
pyplot.legend()
pyplot.show()

#Percentage of data filled using Kriging
import os
import pandas

total_missing = 0
missing_filled = 0

root_1 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion2/'
root_2 = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion3/'

L = os.listdir(root_1)
for i in range(len(L)):
    print(L[i])
    table1 = pandas.read_csv(root_1 + L[i], index_col=0)
    table2 = pandas.read_csv(root_2 + L[i], index_col=0)
    subtable1 = table1[table1['vals']==99999.0]
    subtable2 = table2[table2['vals']==99999.0]
    total_missing += subtable1.shape[0]
    missing_filled += subtable1.shape[0] - subtable2.shape[0]

print(missing_filled/total_missing)

#QUALITY INDEX
import os
import pandas
import datetime
import numpy

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2019,12,31)
total_days = (end-start).days 
L = os.listdir(root_in)

#Percentage of days in the 1980 to 2019 range
P = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    boolean = (table['time']>=start) & (table['time']<=end)
    subtable = table[boolean]
    if subtable.empty:
        P.append(0)
        continue
    days = 0.0
    subtable.reset_index(drop=True, inplace=True)
    for j in range(subtable.shape[0]):
        if subtable.loc[j,'vals'] < 99999.0: days += 1.0
    P.append(days/total_days*100)
    print(str(i) + ': ' +L[i])

P = numpy.asarray(P)
P = P.mean()
print(P)

#Percentage of gaps
Qgaps = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    ngap = 0
    max_length = 0
    length = 0
    n = table.shape[0]
    for j in range(n):
        if table.loc[j,'vals'] == 99999.0: 
            ngap += 1.0
            length += 1
        else: length = 0
        if length > max_length: max_length = length
    Qgaps.append(100 - 100*((2*ngap+max_length)/n))
    print(str(i) + ': ' +L[i])
    
Qgaps = numpy.asarray(Qgaps)
Qgaps = Qgaps.mean()
print(Qgaps)

#percentage of non-null accumulated precipitation on each month
Qnon_null = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    current_year = first_year
    m0 = numpy.zeros(12)
    m = numpy.zeros(12)
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if subtable.empty:
            current_year += 1
            continue
        for j in range(1,13):
            month_table = subtable[pandas.DatetimeIndex(subtable['time']).month == j]
            if month_table.empty: continue
            month_table.reset_index(drop=True, inplace=True)
            cont = 0
            for k in range(month_table.shape[0]):
                val = month_table.loc[k,'vals']
                if 0 < val < 99999.0: break
                else: cont += 1
            if cont == month_table.shape[0]: m0[j-1] += 1
            m[j-1] += 1
        current_year += 1
    m0 = m0.sum()
    m = m.sum()
    Qnon_null.append(100 - 100*m0/m)    

Qnon_null = numpy.asarray(Qnon_null)
Qnon_null = Qnon_null.mean()
print(Qnon_null)

#percentaje Coefficient of variation
QCV = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    current_year = first_year
    ni = [ [] for _ in range(7) ]
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if subtable.empty:
            current_year += 1
            continue
        for j in range(7):
            week_day_table = subtable[pandas.DatetimeIndex(subtable['time']).weekday== j]
            if week_day_table.empty: continue
            week_day_table.reset_index(drop=True, inplace=True)
            cont = 0
            for k in range(week_day_table.shape[0]):
                val = week_day_table.loc[k,'vals']
                if 1 < val < 99999.0: 
                    ni[j].append(val)    
        current_year += 1
    
    CV = 0
    for j in range(7):
        ni[j] = numpy.asarray(ni[j])
        Q1 = numpy.percentile(ni[j],25)
        Q3 = numpy.percentile(ni[j],75)
        CV += (Q3-Q1) / (Q3+Q1) 
    CV /= 7
    QCV.append(100 - 100*CV)    

QCV = numpy.asarray(QCV)
QCV = QCV.mean()
print(QCV)

#Percentage of outliers
Qoutliers = []
for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    n = table.shape[0]
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['DGD'] > 0)
    cond = cond | (table['Persistencia'] > 0)
    cond = cond | (table['PEDSP'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    if subtable.empty:
       Qoutliers.append(100)
       continue
    nout = subtable.shape[0]
    Qoutliers.append(100 - 100*(nout/n))
    print(str(i) + ': ' +L[i])
    
Qoutliers = numpy.asarray(Qoutliers)
Qoutliers = Qoutliers.mean()
print(Qoutliers)

Q = (P+Qgaps+Qnon_null+QCV+Qoutliers)/5

#QUALITY INDEX 2, Figure 4b
import os
import pandas
import datetime
import numpy
from matplotlib import pyplot

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2019,12,31)
total_days = (end-start).days 
L = os.listdir(root_in)

Qrange=numpy.arange(40,101,5)
nstations = numpy.zeros(Qrange.size)

for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    #Percentage of days in the 1980 to 2019 range
    boolean = (table['time']>=start) & (table['time']<=end)
    subtable = table[boolean]
    if subtable.empty: P = 0
    else:
        days = 0.0
        subtable.reset_index(drop=True, inplace=True)
        for j in range(subtable.shape[0]):
            if subtable.loc[j,'vals'] < 99999.0: days += 1.0
        P = days/total_days*100
    #Percentage of gaps
    ngap = 0
    max_length = 0
    length = 0
    n = table.shape[0]
    for j in range(n):
        if table.loc[j,'vals'] == 99999.0: 
            ngap += 1.0
            length += 1
        else: length = 0
        if length > max_length: max_length = length
    Qgaps = 100 - 100*((2*ngap+max_length)/n)
    #percentage of non-null accumulated precipitation on each month
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    current_year = first_year
    m0 = numpy.zeros(12)
    m = numpy.zeros(12)
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if subtable.empty:
            current_year += 1
            continue
        for j in range(1,13):
            month_table = subtable[pandas.DatetimeIndex(subtable['time']).month == j]
            if month_table.empty: continue
            month_table.reset_index(drop=True, inplace=True)
            cont = 0
            for k in range(month_table.shape[0]):
                val = month_table.loc[k,'vals']
                if 0 < val < 99999.0: break
                else: cont += 1
            if cont == month_table.shape[0]: m0[j-1] += 1
            m[j-1] += 1
        current_year += 1
    m0 = m0.sum()
    m = m.sum()
    Qnon_null = 100 - 100*m0/m
    #percentaje Coefficient of variation
    current_year = first_year
    ni = [ [] for _ in range(7) ]
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if subtable.empty:
            current_year += 1
            continue
        for j in range(7):
            week_day_table = subtable[pandas.DatetimeIndex(subtable['time']).weekday== j]
            if week_day_table.empty: continue
            week_day_table.reset_index(drop=True, inplace=True)
            cont = 0
            for k in range(week_day_table.shape[0]):
                val = week_day_table.loc[k,'vals']
                if 1 < val < 99999.0: 
                    ni[j].append(val)    
        current_year += 1
    CV = 0
    for j in range(7):
        ni[j] = numpy.asarray(ni[j])
        Q1 = numpy.percentile(ni[j],25)
        Q3 = numpy.percentile(ni[j],75)
        CV += (Q3-Q1) / (Q3+Q1) 
    CV /= 7
    QCV = 100 - 100*CV    
    #Percentage of outliers
    n = table.shape[0]
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['DGD'] > 0)
    cond = cond | (table['Persistencia'] > 0)
    cond = cond | (table['PEDSP'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    if subtable.empty: Qoutliers=100
    else:
        nout = subtable.shape[0]
        Qoutliers = 100 - 100*(nout/n)
    Q = (P+Qgaps+Qnon_null+QCV+Qoutliers)/5 
    for j in range(Qrange.size):
        if Qrange[j]-2.5 <= Q < Qrange[j]+2.5:
            nstations[j] += 1
            continue
    print(str(i) + ': ' +L[i])

nstations = (nstations / len(L))*100
    
pyplot.bar(Qrange, nstations)
pyplot.xlabel('Quality Index (Q)')
pyplot.ylabel('Percentage of stations (%)')
pyplot.title('Precipitation')
pyplot.show()

#Figure 5b
import os
import pandas
import datetime
import numpy
from matplotlib import pyplot

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2019,12,31)
total_days = (end-start).days 
L = os.listdir(root_in)

dates = pandas.date_range(start,end)
nstations1 = numpy.zeros(dates.size)
nstations2 = numpy.zeros(dates.size)
nstations3 = numpy.zeros(dates.size)

for i in range(len(L)):
    table = pandas.read_csv(root_in + L[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    print(str(i) + ': ' +L[i])
    #Percentage of days in the 1980 to 2019 range
    boolean = (table['time']>=start) & (table['time']<=end)
    subtable0 = table[boolean]
    if subtable0.empty: P = 0
    else:
        days = 0.0
        subtable0.reset_index(drop=True, inplace=True)
        for j in range(subtable0.shape[0]):
            if subtable0.loc[j,'vals'] < 99999.0: days += 1.0
        P = days/total_days*100
    #Percentage of gaps
    ngap = 0
    max_length = 0
    length = 0
    n = table.shape[0]
    for j in range(n):
        if table.loc[j,'vals'] == 99999.0: 
            ngap += 1.0
            length += 1
        else: length = 0
        if length > max_length: max_length = length
    Qgaps = 100 - 100*((2*ngap+max_length)/n)
    #percentage of non-null accumulated precipitation on each month
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    current_year = first_year
    m0 = numpy.zeros(12)
    m = numpy.zeros(12)
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if subtable.empty:
            current_year += 1
            continue
        for j in range(1,13):
            month_table = subtable[pandas.DatetimeIndex(subtable['time']).month == j]
            if month_table.empty: continue
            month_table.reset_index(drop=True, inplace=True)
            cont = 0
            for k in range(month_table.shape[0]):
                val = month_table.loc[k,'vals']
                if 0 < val < 99999.0: break
                else: cont += 1
            if cont == month_table.shape[0]: m0[j-1] += 1
            m[j-1] += 1
        current_year += 1
    m0 = m0.sum()
    m = m.sum()
    Qnon_null = 100 - 100*m0/m
    #percentaje Coefficient of variation
    current_year = first_year
    ni = [ [] for _ in range(7) ]
    while current_year <= last_year:
        subtable = table[pandas.DatetimeIndex(table['time']).year == current_year]
        if subtable.empty:
            current_year += 1
            continue
        for j in range(7):
            week_day_table = subtable[pandas.DatetimeIndex(subtable['time']).weekday== j]
            if week_day_table.empty: continue
            week_day_table.reset_index(drop=True, inplace=True)
            cont = 0
            for k in range(week_day_table.shape[0]):
                val = week_day_table.loc[k,'vals']
                if 1 < val < 99999.0: 
                    ni[j].append(val)    
        current_year += 1
    CV = 0
    for j in range(7):
        ni[j] = numpy.asarray(ni[j])
        Q1 = numpy.percentile(ni[j],25)
        Q3 = numpy.percentile(ni[j],75)
        CV += (Q3-Q1) / (Q3+Q1) 
    CV /= 7
    QCV = 100 - 100*CV    
    #Percentage of outliers
    n = table.shape[0]
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['DGD'] > 0)
    cond = cond | (table['Persistencia'] > 0)
    cond = cond | (table['PEDSP'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable1 = table[cond]
    if subtable1.empty: Qoutliers=100
    else:
        nout = subtable1.shape[0]
        Qoutliers = 100 - 100*(nout/n)
    Q = (P+Qgaps+Qnon_null+QCV+Qoutliers)/5 
    #Selected Q, P, Qoutliers
    good_quality = 1
    if Q >= 80 and P >= 80 and Qoutliers >= 94: good_quality = 3
    elif Q >= 70 and P >= 70 and Qoutliers >= 88: good_quality = 2 
    if not(subtable0.empty):
        for j in range(subtable0.shape[0]):
            if subtable0.loc[j,'vals'] < 99999.0:
                date = subtable0.loc[j,'time']
                pos = dates.get_loc(date)
                if good_quality == 3: nstations1[pos] += 1
                elif good_quality == 2: nstations2[pos] += 1
                else: nstations3[pos] += 1

pyplot.plot(dates, nstations1, 'r', label='Q>=80%, completeness>=80%, outliers<=6%')
pyplot.plot(dates, nstations2, 'b', label='Q>=70%, completeness>=70%, outliers<=12%')
pyplot.plot(dates, nstations3, 'g', label='Q<70% or completeness<70% or outliers>12%')
pyplot.xlabel('Dates')
pyplot.ylabel('Stations')
pyplot.title('Precipitation')
pyplot.legend()
pyplot.show() 

#Probability false negatives 
import pandas
from pandas.tseries.offsets import DateOffset
from calendar import monthrange
import os
import numpy
import math
import scipy.stats as stats
from pyproj import Proj
from random import seed
from random import random
from random import randint
seed(1)

def get_station(file):
    file_split = os.path.splitext(file)
    tokenize = file_split[0].split('_')
    return tokenize[0]

def getWindowDRIGamma(table,date):
    current_month = date.month
    first_year = table.loc[0,'time'].year
    last_year = table.loc[table.shape[0]-1,'time'].year
    W = []
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
                W.append(value)
        current_year += 1
                
    W = numpy.asarray(W)
    return W

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

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion2/'

p = Proj(proj='utm',zone=18,ellps='WGS84')
CNE = pandas.read_excel('D:/Escritorio/Corpoica/Calidad de datos/CNE_IDEAM.xls',
                        index_col=0, dtype={'CODIGO': str})
L = os.listdir(root_in)

Prcpmax = 200
Prcpmin = 0

selected_files = []
for i in range(100):
    selected_files.append(L[randint(0,len(L)-1)])

probability_false_negatives = []
for i in range(len(selected_files)):
    table = pandas.read_csv(root_in + selected_files[i], index_col=0)
    table['time'] = pandas.to_datetime(table['time'])
    station = get_station(selected_files[i])
    print(str(i) + ': ' +selected_files[i])
    row = CNE[CNE['CODIGO'] == station]
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
                station2 = get_station(L[j])
                row2 = CNE[CNE['CODIGO'] == station2]
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
    subtable = table[(table['vals'] > 1) & (table['vals'] < 99999.0)].copy()
    subtable.reset_index(drop=True, inplace=True)
    subtable.insert(subtable.shape[1],'Suspicious',0)
    length = subtable.shape[0]
    index_list = []
    for j in range(200):
        index = randint(0,length-1)
        index_list.append(index)
        val = subtable.loc[index,'vals']
        multiplier = 2.0 + random()*18
        new_val = multiplier*val
        subtable.loc[index,'vals'] = new_val
        if new_val > Prcpmax or new_val < Prcpmin:
            subtable.loc[index,'Suspicious'] = 0.1667*1.44
    for j in range(200):        
        index = index_list[j]
        date = subtable.loc[index,'time']
        W = getWindowDRIGamma(subtable, date)        
        if W.size >= 10:
            ################DRI#################################
            ir = numpy.percentile(W,75) -  numpy.percentile(W,25) #Interquartil range
            PS = numpy.percentile(W,75) + 3 * ir
            ################DGD#################################
            alpha,loc,beta = stats.gamma.fit(W)
            Qp = stats.gamma.ppf(0.995,alpha,loc=loc,scale=beta)
            ####################################################
            value = subtable.loc[index,'vals']
            if value > PS: subtable.loc[index,'Suspicious'] += 0.1667*0.72
            if value > Qp: subtable.loc[index,'Suspicious'] += 0.1667*1.44
    if len(neighbors) >= 2:
       for j in range(200):        
           index = index_list[j]
           date = subtable.loc[index,'time']
           val = subtable.loc[index,'vals']
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
                 if MATD>U: subtable.loc[index,'Suspicious'] += 0.1667*1.44
    table_suspicious = subtable[subtable['Suspicious']>0]
    table_error = subtable[subtable['Error']>0]
    
    prob_false_negatives = (200-table_suspicious.shape[0])/200 
    print(prob_false_negatives)
    probability_false_negatives.append(prob_false_negatives)
    
probability_false_negatives = numpy.asarray(probability_false_negatives)

print('probability false negatives: '+ str(probability_false_negatives.mean())) 

#probability false positives
import os
import pandas

root_in = 'D:/Escritorio/Corpoica/Calidad de datos/IDEAM/Precipitacion1/'
L = os.listdir(root_in)

cont = 0.0
cont2 = 0.0

for i in range(len(L)):
    print(str(i) + ': ' +L[i])
    table = pandas.read_csv(root_in + L[i], index_col=0)
    cont += table.shape[0]
    cond = table['Extremes'] > 0
    cond = cond | (table['DRI'] > 0)
    cond = cond | (table['DGD'] > 0)
    cond = cond | (table['Persistencia'] > 0)
    cond = cond | (table['PEDSP'] > 0)
    cond = cond | (table['Spatial Corroboration'] > 0)
    subtable = table[cond]
    cont2 += subtable.shape[0]        

probability_false_positives = cont2/cont
print('probability false positives: '+ str(probability_false_positives)) 