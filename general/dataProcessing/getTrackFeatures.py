import requests
import base64
import time
import json
import pickle
import numpy as np
import getToken

# define progress file name

# define progress functions
def load_progress():
    progress_file = 'progress/progress.json'
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    except FileNotFoundError:
        progress = 0
    return progress

def save_progress(progress):
    progress_file = 'progress/progress.json'
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def load_itemFeatures(numberOfSongs):
    itemFeaturesfile = 'data/itemFeatures.npy'
    try:
        itemFeatures = np.load(itemFeaturesfile)
    except FileNotFoundError:
        itemFeatures = np.zeros((numberOfSongs, 16))
    return itemFeatures

def getTrackFeatures(header):
    albumIdToIndex = pickle.load(open('data/albumIdToIndex.p','rb'))
    artistIdToIndex = pickle.load(open('data/artistIdToIndex.p','rb'))
    SongIdToIndex = pickle.load(open('data/SongIdToIndex.p','rb'))
    itemFeatures = load_itemFeatures(len(SongIdToIndex))
    
    # load progress
    progress = load_progress()
    print(progress)
    start_index = progress
    for trackInterval in range(start_index, len(SongIdToIndex), 50): 
        intStart, intEnd = trackInterval, trackInterval+50
        print(intStart,intEnd)
        tracks = list(SongIdToIndex.keys())[intStart:intEnd]
        tracks = ','.join(tracks)
        
        # Get the track features for a track
        try : 
            respTrack = requests.get(f'https://api.spotify.com/v1/tracks?ids={tracks}', headers=header)
            respTrackAF = requests.get(f'https://api.spotify.com/v1/audio-features?ids={tracks}', headers=header)
            respTrack.raise_for_status()
            respTrackAF.raise_for_status()
        except requests.exceptions.RequestException as e:
            #store the necessary data and exit the loop
            with open('data/itemFeatures.npy', 'wb') as f:
                 np.save(f, itemFeatures)
            # if there's an error, store the necessary data and exit the loop
            save_progress(intStart)
            print("Error in getting track features, progress stored in progress.json and item features in itemFeatures.npy")
            return False
        track = respTrack.json()
        trackAF = respTrackAF.json()
        for i in range(len(track['tracks'])):
            trackFeatures = []
            if track['tracks'][i] is None:
                trackFeatures.append(-1)
                trackFeatures.append(-1)
                trackFeatures.append(-1)
                trackFeatures.append(-1)
            else:  
                try:
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
            try : 
                trackFeatures.append(trackAF['audio_features'][i]['danceability'])
            except:
                trackFeatures.append(-1)
            try :
                trackFeatures.append(trackAF['audio_features'][i]['energy'])
            except:
                trackFeatures.append(-1)
            try :
                trackFeatures.append(trackAF['audio_features'][i]['key'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['loudness'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['mode'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['speechiness'])
            except:
                trackFeatures.append(-1)           
            try:
                trackFeatures.append(trackAF['audio_features'][i]['acousticness'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['instrumentalness'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['liveness'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['valence'])
            except:
                trackFeatures.append(-1)
            try:    
                trackFeatures.append(trackAF['audio_features'][i]['tempo'])
            except:
                trackFeatures.append(-1)
            try:
                trackFeatures.append(trackAF['audio_features'][i]['time_signature'])
            except:
                trackFeatures.append(-1)
            # Append the features to the matrix
            itemFeatures[intStart+i] = trackFeatures
        time.sleep(0.2) # Sleep
   # with open('data/itemFeatures.npy', 'wb') as f:
    np.save('data/itemFeatures', itemFeatures)
    return True


print(getTrackFeatures(getToken.getSpotifyToken()))