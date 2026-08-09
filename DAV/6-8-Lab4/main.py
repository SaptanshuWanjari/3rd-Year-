from pprint import pprint

import requests

city_name = input("Enter the city name: ")
api_key = "90221a86b39061a4887a3d9b5491c1b2"

url = "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=imperial".format(
    city_name, api_key
)

get_info = requests.get(url)
data = get_info.json()
pprint(data)


print("\nEnvironmental ")
lat = 21.1458
lon = 79.0882

url = "https://api.openweathermap.org/data/2.5/air_pollution?lat={}&lon={}&appid={}".format(
    lat, lon, api_key
)

get_info = requests.get(url)
data = get_info.json()

pprint(data)
