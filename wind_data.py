'''
import pyowm

def get_own():
    return pyowm.OWM('15fc8a92ee87b30b3c6ea450210bac94')


def observation_at_coord(owned, lat, long):
    return owned.weather_at_coords(lat, long)


def get_vector_components(wind_outside):
    deg = wind_outside['deg']
    speed = wind_outside['speed']
    return speed * cos(deg * pi / 180), speed * sin(deg * pi / 180)


if __name__=='__main__':
    own = get_own()
    observation = observation_at_coord(own, 29.729587, -95.571968)
    print(observation)
    weather = observation.get_weather()
    print(weather)
    wind = weather.get_wind()
    print(wind)
'''

from requests import get
from math import sin, cos, pi
import json
from pprint import pprint

url = 'https://apex.oracle.com/pls/apex/raspberrypi/weatherstation/getallstations'
stations = get(url).json()['items']


def closest(lat, long):
    lat = float(lat)
    long = float(long)
    dist = 100000
    station_id = -1
    for station in stations:
        if (station['weather_stn_lat'] - lat) ** 2 + (station['weather_stn_long'] - long) ** 2 < dist:
            dist = (station['weather_stn_lat'] - lat) ** 2 + (station['weather_stn_long'] - long) ** 2
            station_id = station['weather_stn_id']
    return station_id


def get_wind_components(lat, long):
    lat = str(lat)
    long = str(long)
    stn_id = str(closest(lat, long))
    url_id = 'https://apex.oracle.com/pls/apex/raspberrypi/weatherstation/getlatestmeasurements/' + stn_id
    weather = get(url).json()['items']
    wind_speed = weather['wind_speed']
    wind_direction = weather['wind_direction']
    return wind_speed * cos(wind_direction * pi / 180), wind_speed * sin(wind_direction * pi / 180)


def get_coord(lat, long, time):
    lat = float(lat)
    long = float(long)
    wind_x, wind_y = get_wind_components(lat, long)
    long_f = long_hcc + wind_x * (time * 24)
    lat_f = lat_hcc + wind_y * (time * 24)
    return lat_f, long_f

if __name__=='__main__':
    lat_hcc = 29.729587
    long_hcc = -95.571968
    time = 0.5
    lat_hcc_f, long_hcc_f = get_coord(lat_hcc, long_hcc, time)
    print()