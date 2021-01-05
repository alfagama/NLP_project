#   https://plotly.com/python/choropleth-maps/
import plotly
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go

#   ###############################################
#   insert here df with 2 columns [country][counts]
#   ###############################################
loc_dic = {"United States": 28950, "United Kingdom": 16886, "Canada": 7511, "Panama": 615, "India": 7668,
           "Costa Rica": 547, "New Caledonia": 782, "France": 563, "Malaysia": 243, "Australia": 3079, "Gabon": 695,
           "Israel": 1584, "Mongolia": 352, "Morocco": 1237, "Germany": 663, "Moldova: Republic of": 557, "Lesotho": 4,
           "China": 401, "Niger": 124, "Argentina": 723, "Brazil": 993, "Mexico": 58, "South Africa": 1148,
           "Colombia": 677, "Italy": 252, "Azerbaijan": 391, "Tunisia": 499, "Montenegro": 48,
           "Holy See (Vatican City State)": 467, "Lao People's Democratic Republic": 182, "Ireland": 910, "Macao": 352,
           "Netherlands": 674, "Pakistan": 678, "Saudi Arabia": 71, "Jamaica": 95, "Seychelles": 228, "Jordan": 193,
           "Portugal": 183, "Bangladesh": 226, "Nigeria": 520, "Cayman Islands": 160, "Ethiopia": 12, "Belgium": 287,
           "Dominican Republic": 49, "Albania": 165, "Iraq": 6, "Philippines": 232, "Uganda": 51, "Finland": 84,
           "Switzerland": 241, "Mauritius": 38, "Russian Federation": 21, "Indonesia": 172, "Norway": 104,
           "United Arab Emirates": 300, "Georgia": 130, "Belarus": 115, "Puerto Rico": 29, "Spain": 251,
           "Antarctica": 17, "Malta": 70, "Qatar": 239, "Bahrain": 51, "Singapore": 136, "Hong Kong": 168,
           "Bulgaria": 19, "Venezuela: Bolivarian Republic of": 22, "Oman": 60, "Jersey": 63, "Sri Lanka": 79,
           "Zimbabwe": 24, "Kenya": 214, "Turkey": 88, "Greece": 127, "Austria": 47, "New Zealand": 197, "Cyprus": 47,
           "Egypt": 58, "The Netherlands": 18, "Japan": 125, "Hungary": 45, "Lebanon": 93,
           "Taiwan: Province of China": 10, "Rwanda": 9, "Denmark": 55, "Korea: Republic of": 41, "Czech Republic": 1,
           "Guyana": 3, "Poland": 54, "Sudan": 37, "Thailand": 54, "Haiti": 7, "Ghana": 35, "Guam": 6, "Sweden": 116,
           "Lithuania": 8, "Estonia": 8, "Ukraine": 16, "North Macedonia": 6, "Serbia": 42, "Myanmar": 10,
           "Bermuda": 19, "Botswana": 9, "Turkmenistan": 11, "Republic of the Philippines": 8, "Somalia": 4,
           "Slovenia": 16, "Slovakia": 25, "Liberia": 21, "Uruguay": 27, "Bosnia and Herzegovina": 9, "Zambia": 5,
           "Cambodia": 9, "Nepal": 43, "Guatemala": 4, "Algeria": 3, "Cameroon": 8, "Malawi": 5,
           "Iran: Islamic Republic of": 15, "Kuwait": 30, "Yemen": 1, "Gibraltar": 5, "Ecuador": 6,
           "Congo: The Democratic Republic of the": 14, "Vietnam": 1, "Barbados": 3, "Czechia": 27,
           "People's Republic of China": 4, "Mozambique": 2, "Bhutan": 1, "Chile": 13, "Honduras": 5, "Monaco": 13,
           "Romania": 13, "Afghanistan": 25, "Kingdom of Saudi Arabia": 14, "Congo": 4, "Luxembourg": 14,
           "American Samoa": 2, "Croatia": 19, "Bahamas": 3, "Peru": 10, "Togo": 1, "Fiji": 10, "Iceland": 15,
           "Russia": 1, "Latvia": 2, "Eritrea": 2, "Greenland": 1, "Taiwan": 1, "Senegal": 2, "Eswatini": 2,
           "Kazakhstan": 1, "French Guiana": 10, "Palau": 1, "Nicaragua": 3, "Palestine: State of": 5, "Madagascar": 1,
           "Aruba": 1, "Namibia": 4, "Uzbekistan": 3, "Cuba": 3, "Republic of Croatia": 1,
           "Libya": 3, "Saint Helena: Ascension and Tristan da Cunha": 1, "Maldives": 3, "Viet Nam": 5, "Guernsey": 3,
           "Syrian Arab Republic": 4, "Armenia": 2, "": 3, "Sierra Leone": 5, "Tanzania": 1, "Trinidad and Tobago": 2,
           "Timor-Leste": 3, "Tonga": 2, "Turks and Caicos Islands": 1, "Belize": 2, "Hashemite Kingdom of Jordan": 1,
           "Liechtenstein": 1, "South Sudan": 1, "Niue": 1, "Pitcairn": 1, "Suriname": 1, "Grenada": 1, "Kosovo": 2,
           "Republic of Serbia": 1, "Mali": 1, "Cote d'Ivoire": 1, "Northern Mariana Islands": 1}

#   ###############################################
#   df with countries and total number of tweets in each
#   ###############################################
df_counts = pd.DataFrame(loc_dic.items(), columns=['country', 'counts'])

#   ###############################################
#   merge df with gapminder on 'country'
#   ###############################################
gapminder = px.data.gapminder().query("year==2007")
df_countries = pd.DataFrame(gapminder)
df_merged = pd.merge(df_countries, df_counts, on='country')

#   ###############################################
#   plot 1 - the one Alexia did not like
#   ###############################################
# fig = px.choropleth(df_merged, locations="iso_alpha",
#                     color="counts",
#                     hover_name="country",
#                     color_continuous_scale=px.colors.sequential.Plasma)
#
# plotly.offline.plot(fig, filename='worldmap.html')
# fig.show()

#   ###############################################
#   colorscale - insert whatever colors you want in order highest count to lowest count
#   ###############################################
# colorscale = ["#000080", "#071317", "#0c2026", "#102c35", "#153944", "#1e5262", "#276b80", "#30839f", "#399cbd"]
# colorscale = ["#00308F", "#0643A5", "#0C56BC", "#126AD2", "#187DE9", "#1E90FF"]
colorscale = ["#00308F", "#0643A5", "#126AD2", "#1E90FF"]

#   ###############################################
#   plot 2 - the one Alexia liked more
#   ###############################################
fig = go.Figure(data=go.Choropleth(
    locations=df_merged['iso_alpha'],
    z=df_merged['counts'],
    text=df_merged['country'],
    # colorscale='Blues',
    colorscale=colorscale,
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix='tweets: ',
    # colorbar_title='GDP<br>Billions US$',
    colorbar_title='Tweets',
))

fig.update_layout(
    title_text='2020-12-03 - 2020-12-27 #vaccine & #covid19 Tweets',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    # annotations=[dict(
    #     x=0.55,
    #     y=0.1,
    #     xref='paper',
    #     yref='paper',
    #     # text='Source: <a href="https:">\
    #     #     something here</a>',
    #     showarrow=False
    # )]
)
plotly.offline.plot(fig, filename='Locations/worldmap_in_html3.html')
fig.show()
