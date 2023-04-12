import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import time
import json
import pickle
import scipy.sparse as sp
import pickle
import DataLoadEval
from sklearn.model_selection import train_test_split


def dataLoader():
    # Initialize variables 
    numUsers = 1000000 # From spot_mpd/data/stats.txt
    numTracks = 2262292 # From spot_mpd/data/stats.txt
    songIdToIndex = {} # Dictionary to map song IDs to matrix indices
    albumIdToIndex = {} # Dictionary to map album IDs to assign numeric values to ids
    artistIdToIndex = {} # Dictionary to map artist IDs to assign numeric values to ids

    # Create empty matrix trainMatrix
    matrixTrain = sp.lil_matrix((numUsers, numTracks), dtype=np.int8) 
    matrixValid = sp.lil_matrix((numUsers, numTracks), dtype=np.int8)
    matrixTest = sp.lil_matrix((numUsers, numTracks), dtype=np.int8)

    #List of lists structure to build spars UxI matrix incrementally 
    
    # Load the data from the JSON files
    for i in range(0,1000000-999,1000):
        i_min, i_max = i, i+999 # 0-999, 1000-1999, 2000-2999, ...
        print(f'Loading data from {i_min} to {i_max}...')
        playlistInterval = str(i_min) + '-' + str(i_max)
        playlistsInInterval = json.load(open(f'spot_mpd/data/mpd.slice.{playlistInterval}.json'))
        for playlist in playlistsInInterval['playlists']:
            trackIds = []
            userId = playlist['pid'] # Each Playlist has a unique ID hence unique UserID
            for track in playlist['tracks']: # Each Playlist has a list of tracks
                trackId = track['track_uri'].split(':')[2] # Each track has a unique ID
                albumId = track['album_uri'].split(':')[2] # Each album has a unique ID
                artistId = track['artist_uri'].split(':')[2] # Each artist has a unique ID

                if trackId not in songIdToIndex: # Map each track ID to a unique index to build the matrix
                # Assign a new index to the song ID if it has not been encountered before
                    songIdToIndex[trackId] = len(songIdToIndex)
                
                if albumId not in albumIdToIndex: 
                    albumIdToIndex[albumId] = len(albumIdToIndex)
                if artistId not in artistIdToIndex:
                    artistIdToIndex[artistId] = len(artistIdToIndex)


                trackIds.append(songIdToIndex[trackId])
            
            #split the data into train, validation and test sets associated with the given user
            trainTracks, testValTracks = train_test_split(trackIds, test_size=0.30, random_state=42)
            valTracks, testTracks = train_test_split(testValTracks, test_size=0.5, random_state=42)

            # Set the matrix element to 1 if the user has listened to the song
            matrixTrain[userId, trainTracks] = 1
            matrixValid[userId, valTracks] = 1
            matrixTest[userId, testTracks] = 1


    # Save the matrix as a sparse matrix with scipy sparse library
    sp.save_npz('TrainUserItemMatrix.npz', matrixTrain.tocsr())
    sp.save_npz('ValidationUserItemMatrix.npz', matrixValid.tocsr())
    sp.save_npz('TestUserItemMatrix.npz', matrixTest.tocsr())
    # Save the songIdToIndex dictionary as a pickle file
    pickle.dump(songIdToIndex, open( "SongIdToIndex.p", "wb" ))
    pickle.dump(albumIdToIndex, open( "albumIdToIndex.p", "wb" ))
    pickle.dump(artistIdToIndex, open( "artist IdToIndex.p", "wb" ))
    return True

def main():
    if dataLoader():
        print('Data Loaded!')
        if DataLoadEval.EvaluateUserItemRelation():
            print('Success!')
        else:
            print('Error in Indexes!')
    else:
        print('Error in Loading data!')

if __name__ == '__main__':
    main()
