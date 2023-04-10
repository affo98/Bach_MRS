import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import random


def EvaluateUserItemRelation():
    rowId = random.randint(0,999)
    matrix = sp.load_npz('UserItemMatrix.npz')
    songToIndex = pickle.load(open( "SongIdToIndex.p", "rb" ))

    playlist = matrix[rowId, :].nonzero()[1] # returns indexes of nonzero elements
    tracks = [list(songToIndex.keys())[list(songToIndex.values()).index(i)] for i in playlist]
    
    data = json.load(open('../spot_mpd/data/mpd.slice.0-999.json'))
    data = np.unique([i['track_uri'].split(':')[2] for i in data['playlists'][rowId]['tracks']])
    
    for track, index in zip(tracks, playlist):
        if index != songToIndex[track]:
            return False
    return True