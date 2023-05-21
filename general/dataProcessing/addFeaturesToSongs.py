






import numpy as np
import pickle

def addFeaturesToSongs():
    # load the item features for all songs
    itemFeaturesAll = np.load('../data/itemFeatures.npy') 

    # load the songToIndex for all songs
    songToIndexAll = pickle.load(open('../data/songIndexToId.p','rb'))

    features = ['albumId','artistId','popularity','track_number', 'danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
    

    for song in songToIndexAll.keys():
        row = itemFeaturesAll[song]
        featureDict = dict(zip(features, row))
        metaData = songToIndexAll[song]
        songToIndexAll[song]['metaData'] = metaData
        songToIndexAll[song]['features'] = featureDict

    return pickle.dump(songToIndexAll, open('../data/songIndexToId.p', 'wb')), print('done')

addFeaturesToSongs()