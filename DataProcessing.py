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


def dataLoader():
    # Initialize variables 
    numUsers = 1000000 # From spot_mpd/data/stats.txt
    numTracks = 2262292 # From spot_mpd/data/stats.txt
    songIdToIndex = {} # Dictionary to map song IDs to matrix indices

    # Create empty matrix
    matrix = sp.lil_matrix((numUsers, numTracks), dtype=np.int8) 
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
                if trackId not in songIdToIndex: # Map each track ID to a unique index to build the matrix
                # Assign a new index to the song ID if it has not been encountered before
                    songIdToIndex[trackId] = len(songIdToIndex)

                trackIds.append(songIdToIndex[trackId])
            # Set the matrix element to 1 if the user has listened to the song
            matrix[userId, trackIds] = 1

    # Save the matrix as a sparse matrix with scipy sparse library
    sp.save_npz('UserItemMatrix.npz', matrix.tocsr())
    # Save the songIdToIndex dictionary as a pickle file
    pickle.dump(songIdToIndex, open( "SongIdToIndex.p", "wb" ))
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
