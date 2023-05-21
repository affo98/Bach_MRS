### This function will add the item features to songToIndex, 
### and return the item features as a numpy array for the songs in the data/data100k/ folder

import numpy as np
import pickle

def get_item_features():
    # load the songToIndex for all songs
    songToIndexAll = pickle.load(open('../data/songIndexToId.p','rb'))
    
    # load the songToIndex for the 100k songs
    songToIndex100k = pickle.load(open('../data/songIndexToID100k.pkl','rb'))
    songToIndex100knewUsers = pickle.load(open('../data/data100k/songIndexToID100knewUser.pkl','rb'))
    
    # create a numpy array for the item features for the 100k songs
    itemFeatures100k = np.zeros((len(songToIndex100k), 16))

    new_songToIndex100k = {}
    #add the item features for the 100k songs to the numpy array
    for song in songToIndex100k.keys(): #OG INDEXES [0:201]
        idx100k = songToIndexAll[song]['metaData']['100kIndex']

        songFeatures = list(songToIndexAll[song]['features'].values())
        itemFeatures100k[idx100k] = songFeatures
        #construct 100k songtoindex
        new_songToIndex100k[idx100k] = {}
        new_songToIndex100k[idx100k]['features'] = songToIndexAll[song]['features']
        new_songToIndex100k[idx100k]['metaData'] = songToIndexAll[song]['metaData']
    # save the songToIndex for the 100k songs
    pickle.dump(new_songToIndex100k, open('../data/data100k/new_songIndexToID100k.pkl', 'wb'))
    # np.save('../data/data100k/itemFeatures100k.npy', itemFeatures100k)

    # print('Done with 100k')
    # print('NOW TEST SET')
    
    # create a numpy array for the item features for the 100k testset songs
    itemFeatures100knewUsers = np.zeros((len(songToIndex100knewUsers), 16))
    new_songToIndex100knewUsers = {}
    #add the item features for the 100k testset songs to the numpy array
    for song in songToIndex100knewUsers.keys():
        idx100knewUser = songToIndexAll[song]['metaData']['100kNewUserIndex']
        songFeatures = list(songToIndexAll[song]['features'].values())
        itemFeatures100knewUsers[idx100knewUser] = songFeatures
        #construct 100kNew Users Test set songtoindex
        new_songToIndex100knewUsers[idx100knewUser] = {}
        new_songToIndex100knewUsers[idx100knewUser]['features'] = songToIndexAll[song]['features']
        new_songToIndex100knewUsers[idx100knewUser]['metaData'] = songToIndexAll[song]['metaData']

    # save the songToIndex for the 100k testset songs
    pickle.dump(new_songToIndex100knewUsers, open('../data/data100k/new_songIndexToID100knewUser.pkl', 'wb'))
    #np.save('../data/data100k/itemFeatures100knewUsers.npy', itemFeatures100knewUsers)



    return 'Done'
get_item_features()
