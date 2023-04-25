### This function will add the item features to songToIndex, 
### and return the item features as a numpy array for the songs in the data/data100k/ folder

import numpy as np
import pickle

def get_item_features():
    # load the songToIndex for all songs
    songToIndexAll = pickle.load(open('../data/songIndexToId.p','rb'))
    
    # load the songToIndex for the 100k songs
    songToIndex100k = pickle.load(open('../data/data100k/songIndexToID100k.pkl','rb'))
    songToIndex100knewUsers = pickle.load(open('../data/data100k/songIndexToID100knewUser.pkl','rb'))
    
    # create a numpy array for the item features for the 100k songs
    itemFeatures100k = np.zeros((len(songToIndex100k), 16))

    #add the item features for the 100k songs to the numpy array
    for song in songToIndex100k.keys():
        idx100k = songToIndexAll[song]['metaData']['100kIndex']
        songFeatures = list(songToIndexAll[song]['features'].values())
        itemFeatures100k[idx100k] = songFeatures
        #construct 100k songtoindex
        songToIndex100k[song] = {}
        songToIndex100k[song]['features'] = songToIndexAll[song]['features']
        songToIndex100k[song]['metaData'] = songToIndexAll[song]['metaData']
    # save the songToIndex for the 100k songs
    pickle.dump(songToIndex100k, open('../data/data100k/songIndexToID100k.pkl', 'wb'))
    np.save('../data/data100k/itemFeatures100k.npy', itemFeatures100k)

    print('Done with 100k')
    print('NOW TEST SET')
    
    # create a numpy array for the item features for the 100k testset songs
    itemFeatures100knewUsers = np.zeros((len(songToIndex100knewUsers), 16))
    #add the item features for the 100k testset songs to the numpy array
    for song in songToIndex100knewUsers.keys():
        idx100knewUser = songToIndexAll[song]['metaData']['100kNewUserIndex']
        songFeatures = list(songToIndexAll[song]['features'].values())
        itemFeatures100knewUsers[idx100knewUser] = songFeatures
        #construct 100kNew Users Test set songtoindex
        songToIndex100knewUsers[song] = {}
        songToIndex100knewUsers[song]['features'] = songToIndexAll[song]['features']
        songToIndex100knewUsers[song]['metaData'] = songToIndexAll[song]['metaData']

    # save the songToIndex for the 100k testset songs
    pickle.dump(songToIndex100knewUsers, open('../data/data100k/songIndexToID100knewUser.pkl', 'wb'))
    np.save('../data/data100k/itemFeatures100knewUsers.npy', itemFeatures100knewUsers)



    return 'Done'
get_item_features()
