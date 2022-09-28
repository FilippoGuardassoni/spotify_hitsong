# Billboard API
# See Billboard API instructions in Github - Guo

import billboard
import pandas as pd

# Retrieve the names of existing charts
chart_names = billboard.charts(year_end=True)
print(chart_names)

# Fetching the Hot 100 Year-End chart
chart = billboard.ChartData('hot-100-songs', year=2020)

title = []
artist = []
rank = []
year = []
i = 0

# Loop to save the data in lists
while chart.previousYear:
    for e in chart:
        title.append(e.title)
        artist.append(e.artist)
        rank.append(e.rank)
        year.append(chart.year)
    chart = billboard.ChartData('hot-100-songs', year=chart.previousYear)

# Convert the lists in a dataframe and export to csv
df1 = pd.DataFrame(list(zip(title, artist)), columns=['track_name', 'artist_name'])
df1.to_csv('billboard.csv', header=True)
print(df1)

