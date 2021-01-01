#https://plotly.com/python/choropleth-maps/
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
import geopy
from geopy.geocoders import Nominatim
from pymongo import MongoClient
from collections import Counter
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cmp
from folium.features import DivIcon
from geopy.exc import GeocoderServiceError
from geopy.exc import GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
import plotly
import plotly.express as px

def do_geocode(address, attempt=1, max_attempts=5):
    try:
        return geolocator.geocode(address)
    except GeocoderTimedOut:
        if attempt <= max_attempts:
            return do_geocode(address, attempt=attempt+1)
        raise

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


def get_continent(col):
    try:
        cn_a2_code = country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown'
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown'
    return (cn_a2_code, cn_continent)


#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB

collections = db.collection_names()

#####################################################################

all_countries = []
for collection in collections:
    tweets = db[collection].find().batch_size(10).limit(1000)
    if collection == 'quarantine':
        for tweet in tweets:
            location = db[collection].find_one({'location': tweet["location"]})["location"]
            all_countries.append(str(location))


all_countries = remove_values_from_list(all_countries, 'None')
unique_tweets_frequency = Counter(all_countries)
print(unique_tweets_frequency)

#Creating the COVID-19 DataFrame
df = []
listOfContinents = []
for index, value in enumerate(Counter(all_countries)):
    #print(index)
    #print(value)
    df.append([value, Counter(all_countries)[value]])
    listOfContinents.append(get_continent(value))
df_covid = pd.DataFrame(df, columns = ['country', 'counts'])
print(listOfContinents)


geolocator = Nominatim(user_agent="my_user_agent")
geopy.geocoders.options.default_timeout = 20

#empty map
world_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)

#Setting up the world countries data URL
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
country_shapes = f'{url}/world-countries.json'

step = cmp.StepColormap(
 [ 'red', 'yellow', 'green'],
 vmin=3, vmax=10,
 index=[3, 6, 8, 10],  #for change in the colors, not used fr linear
 caption='Color Scale for Map'    #Caption for Color scale or Legend
)


counter = 0
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1,max_retries=0)
#for each coordinate, create circlemarker of user percent
for index, value in enumerate(Counter(all_countries)):
    if counter < 100000:
        print(value)
        try:
            loc = geolocator.geocode(listOfContinents[index][0] + ',' + listOfContinents[index][1])
            counter = counter + 1
            print(loc)
            if loc != None:
                numberOfTweets = Counter(all_countries)[value]
                lat = loc.latitude
                long = loc.longitude
                radius=10
                popup_text = """Country : {}<br>
                            #of Tweets : {}<br>"""
                popup_text = popup_text.format(value,numberOfTweets)
                folium.map.Marker(location=[lat, long], radius=1,
                                  icon=DivIcon(
                                      icon_size=(20, 20),
                                      icon_anchor=(0, 0),
                                      html='<div style="font-size: 10pt">' + str(numberOfTweets) + '</div>',
                                  )
                                  ).add_to(world_map)
                #folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
            else:
                print("I got None for this country: ", value)
        except GeocoderServiceError:
            print("to efaga")

folium.Choropleth(
        # The GeoJSON data to represent the world country
        geo_data=country_shapes,
        data=df_covid,
        name='choropleth COVID-19',
        # The column aceppting list with 2 value; The country name and  the numerical value
        columns=['country', 'counts'],
        key_on='feature.properties.name',
        fill_color='PuRd',
        nan_fill_color='white',
        highlight="true",
        style_function=lambda feature: {
                'fillColor': 6,
                'color': 'black',       #border color for the color fills
                'weight': 1,            #how thick the border has to be
                'dashArray': '5, 3'  #dashed lines length,space between them
            }
    ).add_to(world_map)

#folium.Tooltip(text="eleni", sticky="false").add_to(world_map)



# save map to html file
step.add_to(world_map)
world_map.save('index.html')


gapminder = px.data.gapminder().query("year==2007")
df_countries = pd.DataFrame(gapminder)
df_merged = pd.merge(df_countries, df_covid, on='country')

fig = px.choropleth(df_merged, locations="iso_alpha",
                    color="counts",
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma)

plotly.offline.plot(fig, filename='worldmap.html')
fig.show()
