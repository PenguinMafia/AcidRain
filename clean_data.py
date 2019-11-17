import pandas as pd
import numpy as np
import re

mmap = {}

mmap['C C BOTANICAL GARDENS, TX US'] = [27.6525, -97.4069, 4.9]
mmap['CULLMAN N.A.H.S., AL US'] = [34.174820, -86.843613, 247.2]
mmap['FREMONT INDIAN S.P., UT US'] = [38.5774, -112.3344, 1804.4]
mmap['HOGSETT R.C. BYRD DAM, WV US'] = [38.68416, -82.18384, 173.7]
mmap['N LAZY H RANCH, AZ US'] = [32.1145231, -110.6911934, 929.6]
mmap['O C FISHER DAM, TX US'] = [31.4833314, -100.4833314, 598.6]
mmap['SQUAW VALLEY G.C., CA US'] = [39.1963, -120.2336, 2447.2]

mmap['HENDERSON 5.9 W, NV US'] = [36.04879, -115.10663, 627.6]
mmap['RESERVE 1 W, NM US'] = [33.7149, -108.7771, 1780.6]
mmap['BRIGHTON 7.7 W, CO US'] = [39.9951, -104.9661, 1564.8]
mmap['LINCOLN 3.9 W, NE US'] = [40.78892, -96.741346, 0]
mmap['DAVENPORT 0.1 W, NE US'] = [40.3128, -97.8135, 508.1]
mmap['KING CITY 4.7 W, MO US'] = [40.05374, -94.613443, 327.1]
mmap['SPICKARD 7 W, MO US'] = [40.2472, -93.7158, 266.7]
mmap['WAVERLY 3 W, MO US'] = [39.2103, -93.5297, 243.8]
mmap['CLAREMORE 6.6 W, OK US'] = [36.329745, -95.73666, 235]
mmap['VALLIANT 3 W, OK US'] = [33.998, -95.1433, 144.8]
mmap['W G HUXTABLE PUMPING PLANT, AR US'] = [34.73528, -90.64861, 57.9]
mmap['MARLTON 1 W, NJ US'] = [39.9, -74.93333, 27.1]
mmap['ELKRIDGE 1.8 W, MD US'] = [39.2021, -76.7843, 103.9]


def days_in_month(numb_month):
    list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list[numb_month - 1]


def to_lat(ray):
    lst = []
    for location in ray:
        lst.append(mmap[location][0])
    return np.array(lst)


def to_long(ray):
    lst = []
    for location in ray:
        lst.append(mmap[location][1])
    return np.array(lst)


def parse_year(ray):
    lst = []
    for date in ray:
        date_split = re.split('-', date)
        lst.append(int(date_split[0]))
    return np.array(lst)


def parse_month_day(ray):
    lst = []
    for date in ray:
        date_split = re.split('-', date)
        number = int(date_split[1]) + (int(date_split[2]) - 1) / days_in_month(int(date_split[1]))
    return np.array(lst)


def get_elevation(ray):
    lst = []
    for location in ray:
        lst.append(mmap[location][2])
    return np.array(lst)


df = pd.read_csv('1947225.csv')
df = df[['STATION', 'NAME', 'DATE', 'PRCP']]
df = df.dropna(subset=['PRCP'])
df = df.reset_index(drop=True)

df['LATITUDE'] = to_lat(np.array(df['NAME']))
df['LONGITUDE'] = to_long(np.array(df['NAME']))

df['YEAR'] = parse_year(np.array(df['DATE']))
df['MONTH'] = parse_month_day(np.array(df['DATE']))
df['ELEVATION'] = get_elevation(np.array(df['NAME']))
df['PRECIPITATION'] = df['PRCP']

df = df[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'MONTH', 'YEAR', 'PRECIPITATION']]

df.to_csv(r'clean_data3.csv')