import pyowm
from math import cos, sin, pi
from pprint import pprint

own = pyowm.OWM('15fc8a92ee87b30b3c6ea450210bac94')


def get_wind_components(lat, long):
    wind_outside = own.weather_at_coords(lat, long).get_weather().get_wind()
    deg = 0
    try:
        deg = wind_outside['deg']
    except KeyError:
        deg = 0
    speed = wind_outside['speed']
    return speed * cos(deg * pi / 180), speed * sin(deg * pi / 180)


def get_coord(lat, long, t):
    wind_x, wind_y = get_wind_components(lat, long)
    long_f = long_hcc + (wind_x / 69.0) * (t / 24)
    lat_f = lat_hcc + (wind_y / 69.0) * (t / 24)
    return lat_f, long_f


if __name__=='__main__':
    lat_hcc = 29.729587
    long_hcc = -95.571968
    lat_hcc, long_hcc = 51.5074, -0.1278

    time = 5
    lat_hcc_f, long_hcc_f = get_coord(lat_hcc, long_hcc, time)
    print(lat_hcc_f, "\t", long_hcc_f)

"""
from requests import get
from math import sin, cos, pi
import json
from pprint import pprint
import pyowm

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
    stn_id = closest(lat, long)
    url_id = 'https://apex.oracle.com/pls/apex/raspberrypi/weatherstation/getlatestmeasurements/' + str(stn_id)
    weather = get(url_id).json()['items']
    wind_speed = weather[0]['wind_speed']
    wind_direction = weather[0]['wind_direction']
    return wind_speed * cos(wind_direction * pi / 180), wind_speed * sin(wind_direction * pi / 180)


def get_coord(lat, long, t):
    wind_x, wind_y = get_wind_components(lat, long)
    long_f = long_hcc + (wind_x / 69.0) * (t * 24)
    lat_f = lat_hcc + (wind_y / 69.0) * (t * 24)
    return lat_f, long_f


if __name__ == '__main__':
    lat_hcc = 29.729587
    long_hcc = -95.571968
    time = 25
    lat_hcc_f, long_hcc_f = get_coord(lat_hcc, long_hcc, time)
    print(lat_hcc_f, long_hcc_f)
"""
