#   https://plotly.com/python/choropleth-maps/
import plotly
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go



# LOAD THE CSVs WITH THE TWEETS COUNT PER COUNTRY
#-------------------------------------------------
covidDF = pd.read_csv("covid_Country_vis.csv")
vaccineDF = pd.read_csv("vaccine_Country_vis.csv")
quarantineDF = pd.read_csv("quarantine_Country_vis.csv")

covidDF = covidDF.drop(columns='Unnamed: 0')
vaccineDF = vaccineDF.drop(columns='Unnamed: 0')
quarantineDF = quarantineDF.drop(columns='Unnamed: 0')

#--------------------------------------------------
#Merge the 3 dataframes into 1 based on the Country and sum the count of the tweets

df_countries_counts = pd.concat([quarantineDF, vaccineDF, covidDF]).groupby(['country']).sum().reset_index()
print(df_countries_counts)

#   ###############################################
#   merge df with gapminder on 'country'
#   ###############################################
gapminder = px.data.gapminder().query("year==2007")
df_countries = pd.DataFrame(gapminder)
df_merged = pd.merge(df_countries, df_countries_counts, on='country')

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
#colorscale = ["#00308F", "#0643A5", "#126AD2", "#1E90FF"]
colorscale = ['#000069','#00006B','#00006D','#00006F','#000171','#000372','#000674','#000876','#000B78','#000E79','#00107B',
'#00137D','#00167F','#001981','#001C82','#001F84','#002286','#002587','#002889','#002B8B','#002F8C','#00328E','#003590',
'#003991','#003C93','#004095','#004396','#004798','#004B99','#004E9B','#00529D','#00569E','#005AA0','#005EA1','#0062A3',
'#0066A4','#006AA6','#006EA7','#0072A9','#0076AA','#007AAC','#007EAD','#0082AF','#0087B0','#008BB2','#008FB3','#0093B4',
'#0098B6','#009CB7','#00A1B9','#00A5BA','#04AABB','#07AFBC','#0BB3BD','#0EB8BE','#12BCBF','#15C0BF','#19C1BD','#1CC2BA',
'#20C3B8','#23C4B6','#27C5B4','#2AC5B2','#2EC6B1','#31C7AF','#35C8AD','#38C9AC','#3CCAAB','#3FCBA9','#43CCA8','#46CDA7','#4ACEA6',
'#4DCFA6','#51D0A5','#55D1A4','#58D2A4','#5CD3A3','#5FD4A3','#63D5A3','#66D6A3','#6AD7A3','#6DD8A3','#71D9A3','#74D9A3',
'#78DAA4','#7BDBA4','#7FDCA5','#82DDA5','#86DEA6','#89DFA7','#8DE0A8','#90E1A9','#94E2AA','#97E3AC','#9BE4AD','#9EE5AE',
'#A2E6B0','#A5E7B2','#A9E8B4','#ACE9B5','#B0EAB7']

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
    colorbar_tickprefix='',
    # colorbar_title='GDP<br>Billions US$',
    colorbar_title='Tweets',
))

fig.update_layout(
    title_text='Number of tweets collected between 06 - 29 December 2020',
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
plotly.offline.plot(fig, filename='testMap.html')
fig.show()
