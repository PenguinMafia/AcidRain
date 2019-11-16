import pandas as pd
import numpy as np
import re

mmap = {}

mmap['C C BOTANICAL GARDENS, TX US'] = [27.6525, -97.4069, 4.9]
mmap['CULLMAN N.A.H.S., AL US'] = [34.174820, -86.843613, ]
mmap['FREMONT INDIAN S.P., UT US'] = [38.5774, -112.3344]
mmap['HOGSETT R.C. BYRD DAM, WV US'] = [38.68416, -82.18384]
mmap['N LAZY H RANCH, AZ US'] = [32.1145231, -110.6911934]
mmap['O C FISHER DAM, TX US'] = [31.4833314, -100.4833314, ]
mmap['SQUAW VALLEY G.C., CA US'] = [39.1963, -120.2336, 2447.2]

print(mmap)

def days_in_month(numb_month):
    list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list[numb_month - 1]

def to_lat(ray):
    lst = []
    for location in ray:
        lst.append(mmap[location][0])
    print(lst)
    return np.array(lst)


def to_long(ray):
    lst = []
    for location in ray:
        lst.append(mmap[location][1])
    return np.array(lst)


def parse_year(ray):
    lst = []
    for date in ray:
        date_split = re.split('/', date)
        lst.append(int(date_split[2]))
    return np.array(lst)


def parse_month_day(ray):
    lst = []
    for date in ray:
        date_split = re.split('/', date)
        number = int(date_split[0]) + (int(date_split[1]) - 1) / days_in_month(int(date_split[0]))
        lst.append(number)
    return np.array(lst)


df = pd.read_csv('1946944.csv')
df = df[['STATION', 'NAME', 'DATE', 'PRCP']]
df = df.dropna(subset=['PRCP'])
df = df.reset_index(drop=True)

df['LAT'] = to_lat(np.array(df['NAME']))
df['LONG'] = to_long(np.array(df['NAME']))

df['YEAR'] = parse_year(np.array(df['DATE']))
df['MONTH'] = parse_month_day(np.array(df['DATE']))

df.to_csv(r'clean_data.csv')