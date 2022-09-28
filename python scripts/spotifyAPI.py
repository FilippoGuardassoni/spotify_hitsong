# Spotify API (spotipy)

# usual imports
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials as SCC
import pandas as pd
import socket

# connect spotipy api with spotify developer web app
SPOTIPY_CLIENT_ID = "..."
SPOTIPY_CLIENT_SECRET = "..."

client_credentials_manager = SCC(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=client_credentials_manager)

# import billboard songs
df1 = pd.read_csv('billboard.csv')
df1 = df1.drop('index', axis=1)

# check for possible synonyms of featuring
def featuring_(row):
    if 'Featuring' in row['artists_name']:
        val = 1
    elif 'Duet With' in row['artists_name']:
        val = 1
    elif 'With' in row['artists_name']:
        val = 1
    elif 'X' in row['artists_name']:
        val = 1
    elif 'x' in row['artists_name']:
        val = 1
    elif '&' in row['artists_name']:
        val = 1
    elif ',' in row['artists_name']:
        val = 1
    elif 'Or' in row['artists_name']:
        val = 1
    elif '+' in row['artists_name']:
        val = 1
    else:
        val = 0
    return val

# split the artist name to consider only the owner of the song
def artist_name_(row):
    none = True
    if 'With' in row['artists_name']:
        val = row['artists_name'].split(" With")[0]
        none = False
    if 'Duet With' in row['artists_name']:
        val=row['artists_name'].split(" Duet With")[0]
        none = False
    if 'X' in row['artists_name']:
        val=row['artists_name'].split(" X ")[0]
        none = False
    if 'x' in row['artists_name']:
        val=row['artists_name'].split(" x")[0]
        none = False
    if '&' in row['artists_name']:
        val=row['artists_name'].split(" &")[0]
        none = False
    if ',' in row['artists_name']:
        val=row['artists_name'].split(",")[0]
        none = False
    if '+' in row['artists_name']:
        val = row['artists_name'].split(" +")[0]
        none = False
    if 'Featuring' in row['artists_name']:
        val=row['artists_name'].split(" Featuring")[0]
        none = False
        if '&' in row['artists_name'] and 'Featuring' in row['artists_name']:
            val = row['artists_name'].split(" &")[0]
            none = False
            if 'Featuring' in val:
                val = row['artists_name'].split(" Featuring")[0]
                none = False
    if 'Or' in row['artists_name']:
        val = row['artists_name'].split(" Or")[0]
        none = False
    if none:
        val = row['artists_name']
    return val

df1['featuring'] = df1.apply(featuring_, axis=1)
df1['artist_name'] = df1.apply(artist_name_, axis=1)

df1.head()

# creation of a new column containing the name of the main artist and the title of the song
df1['search'] = df1['track_name'] + " " + df1['artist_name']

artist_name = []
pop_artist = []
tot_followers = []
track_name = []
avail_mark = []
rel_date = []
pop_track = []
acousticness = []
danceability = []
duration_ms = []
energy = []
instrumentalness = []
liveness = []
loudness = []
speechiness = []
tempo = []
time_signature = []
valence = []

# a function which will be used to search the songs following the search column and to extract -
# - the variables mentioned above
def track_info(search):
    track_ = []
    try:
        track_results = sp.search(q=search, type='track', limit=1, offset=0)
        track_.append(track_results['tracks']['items'])
        track_uri = track_[0][0]['uri']
        audio_features = sp.audio_features(track_uri)[0]

        if audio_features != None:
            # Artist info
            artist_uri = track_[0][0]['album']['artists'][0]['uri']
            artist = sp.artist(artist_uri)
            artist_name.append(artist['name'])
            pop_artist.append(artist['popularity'])
            tot_followers.append(artist['followers']['total'])
            # Track info
            track_name.append(track_[0][0]['name'])
            avail_mark.append(len(track_[0][0]['available_markets']))
            rel_date.append(track_[0][0]['album']['release_date'])
            pop_track.append(track_[0][0]['popularity'])
            acousticness.append(audio_features['acousticness'])
            danceability.append(audio_features['danceability'])
            duration_ms.append(audio_features['duration_ms'])
            energy.append(audio_features['energy'])
            instrumentalness.append(audio_features['instrumentalness'])
            liveness.append(audio_features['liveness'])
            loudness.append(audio_features['loudness'])
            speechiness.append(audio_features['speechiness'])
            tempo.append(audio_features['tempo'])
            time_signature.append(audio_features['time_signature'])
            valence.append(audio_features['valence'])
    except:
        pass

    # Action!
df1['search'].map(track_info)

df1 = pd.DataFrame({'artist_name': artist_name, 'pop_artist': pop_artist, 'tot_followers': tot_followers, 'track_name': track_name,
                    'avail_mark': avail_mark, 'rel_date': rel_date, 'pop_track': pop_track, 'acousticness': acousticness,
                    'danceability': danceability, 'duration_ms': duration_ms, 'energy': energy, 'instrumentalness': instrumentalness,
                    'liveness':liveness, 'loudness': loudness, 'speechiness': speechiness,
                    'tempo': tempo, 'time_signature': time_signature, 'valence': valence
                  })
df1.head()

# recovery of the songs at a rate of 1,000 per year for the period 2007-2019
try:
    track_=[]
    for j in range(2007, 2021, 1):
        year='year:'+str(j)
        for i in range(0, 1000, 50):
            track_results = sp.search(q=year, type='track', limit=50, offset=i)
            track_.append(track_results['tracks']['items'])

    # construction of a data table with the selected fields
    artist_name = []
    pop_artist = []
    tot_followers = []
    track_name = []
    avail_mark = []
    rel_date = []
    pop_track = []
    acousticness = []
    danceability = []
    duration_ms = []
    energy = []
    instrumentalness = []
    liveness = []
    loudness = []
    speechiness = []
    tempo = []
    time_signature = []
    valence = []

    for i in range(0, 260, 1):
        for j in range(0, 50, 1):
            track_uri = track_[i][j]['uri']
            audio_features = sp.audio_features(track_uri)[0]

            if audio_features != None:
                # Artist info
                artist_uri = track_[i][j]['album']['artists'][0]['uri']
                artist = sp.artist(artist_uri)
                artist_name.append(artist['name'])
                pop_artist.append(artist['popularity'])
                tot_followers.append(artist['followers']['total'])

                # Track info
                track_name.append(track_[i][j]['name'])
                avail_mark.append(len(track_[i][j]['available_markets']))
                rel_date.append(track_[i][j]['album']['release_date'])
                pop_track.append(track_[i][j]['popularity'])

                # Track audio info
                acousticness.append(audio_features['acousticness'])
                danceability.append(audio_features['danceability'])
                duration_ms.append(audio_features['duration_ms'])
                energy.append(audio_features['energy'])
                instrumentalness.append(audio_features['instrumentalness'])
                liveness.append(audio_features['liveness'])
                loudness.append(audio_features['loudness'])
                speechiness.append(audio_features['speechiness'])
                tempo.append(audio_features['tempo'])
                time_signature.append(audio_features['time_signature'])
                valence.append(audio_features['valence'])
except socket.timeout:
    pass

df2 = pd.DataFrame(
    {'artist_name': artist_name, 'pop_artist': pop_artist, 'tot_followers': tot_followers, 'track_name': track_name,
     'avail_mark': avail_mark, 'rel_date': rel_date, 'pop_track': pop_track, 'acousticness': acousticness,
     'danceability': danceability, 'duration_ms': duration_ms, 'energy': energy, 'instrumentalness': instrumentalness,
     'liveness': liveness, 'loudness': loudness, 'speechiness': speechiness,
     'tempo': tempo, 'time_signature': time_signature, 'valence': valence
     })

print(df2)

# creation of the hit variable. By definition, all the musics coming from the data table df1 are hits.
df1['hit']=1
df2['hit']=0

# combination of the two data tables
df=pd.concat([df1, df2], ignore_index=True)

# creation of the variable featuring. It is worth 1 if there are guest artists on the music, and 0 otherwise.
def feat_(row):
    if 'feat.' in row['track_name']:
        val = 1
    else:
        val = 0
    return val

df['featuring'] = df.apply(feat_, axis=1)

# check for missing values
df.isnull().sum()

# suppression of the duplicates
df['search'] = df['artist_name'] + " " + df["track_name"]
df=df.drop_duplicates(subset=['search'], keep='first')
df.shape

# retrival of the dates
df['rel_date']=pd.to_datetime(df['rel_date'])
df['rel_month'] = df['rel_date'].apply(lambda m: m.month)
df['rel_day'] = df['rel_date'].apply(lambda d: d.day)
df['week_day_out'] = df['rel_date'].apply(lambda w: w.weekday())
df.shape

# export the dataset
df.to_csv('songs.csv', header=True)
