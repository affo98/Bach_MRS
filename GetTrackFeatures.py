import requests
import base64
import time
import json
import pickle
import getToken
import numpy as np
    

def getTrackFeatures(header):
    albumIdToIndex = pickle.load(open('albumIdToIndex.p','rb'))
    artistIdToIndex = pickle.load(open('artistIdToIndex.p','rb'))
    SongIdToIndex = pickle.load(open('SongIdToIndex.p','rb'))
    itemFeatures = np.zeros((len(SongIdToIndex), 16))

    for trackInterval in range(0,len(SongIdToIndex),50): # The values are sorted by the definition of this specific dictionary, this exploited to construct the matrix directly
        intStart, intEnd = trackInterval, trackInterval+50
        print(intStart, intEnd)
        tracks = list(SongIdToIndex.keys())[intStart:intEnd]
        tracks = ','.join(tracks)
        
        # Get the track features for a track
        track = requests.get(f'https://api.spotify.com/v1/tracks?ids={tracks}', headers=header).json()
        trackAF = requests.get(f'https://api.spotify.com/v1/audio-features?ids={tracks}', headers=header).json()
        for i in range(len(track['tracks'])): # 0-49, 50-99, 100-149, 150-199
            trackFeatures = [] # 16 features
            if track['tracks'][i] is None:
                trackFeatures.append(-1)
                trackFeatures.append(-1)
                trackFeatures.append(-1)
                trackFeatures.append(-1)
            else:
            #print(artistIdToIndex[track['tracks'][i]['artists'][0]['id']])
                try :
                    trackFeatures.append(albumIdToIndex[track['tracks'][i]['album']['id']])
                except:
                    trackFeatures.append(-1) # No Album Id
                try:
                    trackFeatures.append(artistIdToIndex[track['tracks'][i]['artists'][0]['id']])
                except:
                    trackFeatures.append(-1) # No Artist Id
                trackFeatures.append(track['tracks'][i]['popularity'])
                trackFeatures.append(track['tracks'][i]['track_number'])
            
            # Audio Meta Data
            trackFeatures.append(trackAF['audio_features'][i]['danceability'])
            trackFeatures.append(trackAF['audio_features'][i]['energy'])
            trackFeatures.append(trackAF['audio_features'][i]['key'])
            trackFeatures.append(trackAF['audio_features'][i]['loudness'])
            trackFeatures.append(trackAF['audio_features'][i]['mode'])
            trackFeatures.append(trackAF['audio_features'][i]['speechiness'])
            trackFeatures.append(trackAF['audio_features'][i]['acousticness'])
            trackFeatures.append(trackAF['audio_features'][i]['instrumentalness'])
            trackFeatures.append(trackAF['audio_features'][i]['liveness'])
            trackFeatures.append(trackAF['audio_features'][i]['valence'])
            trackFeatures.append(trackAF['audio_features'][i]['tempo'])
            trackFeatures.append(trackAF['audio_features'][i]['time_signature'])
            # Append the features to the matrix indexed by 0,50,100,150 + i in range 0-50, 50-100...
            itemFeatures[intStart+i] = trackFeatures
        time.sleep(1) # Sleep to avoid rate limit
    with open('itemFeatures.npy', 'wb') as f:
        np.save(f, itemFeatures)
    return True

print(getTrackFeatures(getToken.getSpotifyToken()))
