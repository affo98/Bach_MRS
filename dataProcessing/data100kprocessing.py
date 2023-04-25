
# this function will prepare the 100000 data set for the future models
# First it goes through the TrainUserItemMatrix.npz file to extract 100000 random users
# Then it goes through each of the users in the 100000 data set and extracts the 1000 most popular songs.
# While going through each 1000000 user, it also builds a dictionary for albums, artits, songs and songs item features, to look 
# such that that every aspect is optimized for the future models based on 100000 users.
# Finally, it saves the 100000 user data and the lookup dictionaries in the new folder file called data100000.

import numpy as np
import json
import pickle
import random
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

def get100000Users():
    random.seed(42)
    # load the TrainUserItemMatrix.npz file
    UserItemMatrix = sp.load_npz('../data/UserItemMatrix.npz')
    # get the number of users in the TrainUserItemMatrix.npz file
    numberOfUsers = UserItemMatrix.shape[0]
    # get 100000 random users
    randomUsers100k = random.sample(range(0,numberOfUsers), 250000) # add extra users for validation & test, and buffer for no interactions
    newRandomUsers100k = []
    while len(newRandomUsers100k) < 12000:
        randomPlaylist = random.randint(0,numberOfUsers)
        if randomPlaylist not in randomUsers100k:
            newRandomUsers100k.append(randomPlaylist)
        else:
            continue

    # randomUsers Now a list:
    # We only want to get 100000 random users that have interactions i.e. songs in their playlists
    # so we need to get the number of songs per user
    songIndexToID = pickle.load(open('../data/songIndexToId.p', 'rb'))
    songIndexToID100k = {}
    #numTracks100k = 0 # number of total tracks in the 100000 users
    userTrack100kStructure = {}

    # 100000 users
    for user in randomUsers100k: # room for 100 empty playlists
        if len(UserItemMatrix[user].nonzero()[1].tolist()) < 50:
            continue
        else:
            userTrack100kStructure[user] = [] # {u_i: [t_1, t_2, t_3, ...]} using old indices
            #numTracks100k += len(UserItemMatrix[user].nonzero()[1]) 
            for track in UserItemMatrix[user].nonzero()[1]: # get indices of songs in original matrix # 1,2,3,4
                # 42123, 23, 1, 101230, 
                if track not in songIndexToID100k:
                    songIndexToID100k[track] = len(songIndexToID100k) # key= OG index, value = new index
                    songIndexToID[track]['100kIndex'] = songIndexToID100k[track] # minus one, since we already added one
                userTrack100kStructure[user].append(songIndexToID[track]['100kIndex']) # add the track to the user's playlist with the new index
        if len(userTrack100kStructure) == 100000:
            break


    # Compile new users as unseen test set -> cold start problems
    newUserTracksIndexToID100k = {}
    newUserTrack100kStructure = {}
    #newUserNumTracks100k = 0

    for user in newRandomUsers100k: # room for 100 empty playlists
        if len(UserItemMatrix[user].nonzero()[1].tolist()) < 10:
            continue
        else:
            newUserTrack100kStructure[user] = [] # {u_i: [t_1, t_2, t_3, ...]} using new indices
            #newUserNumTracks100k = len(UserItemMatrix[user].nonzero()[1]) 
            for track in UserItemMatrix[user].nonzero()[1].tolist(): # get indices of songs in original matrix
                if track not in newUserTracksIndexToID100k:
                    newUserTracksIndexToID100k[track] = len(newUserTracksIndexToID100k)
                    songIndexToID[track]['100kNewUserIndex'] = newUserTracksIndexToID100k[track]
                newUserTrack100kStructure[user].append(songIndexToID[track]['100kNewUserIndex']) # add the track to the user's playlist with the new index
        if len(newUserTrack100kStructure) == 10000:
            break


    # create a 100000x(number of songs per user * 100000) matrix
    train100k = sp.lil_matrix((len(userTrack100kStructure), len(songIndexToID100k)), dtype=np.int8)
    validation100k = sp.lil_matrix((len(userTrack100kStructure), len(songIndexToID100k)), dtype=np.int8)
    test100k = sp.lil_matrix((len(userTrack100kStructure),len(songIndexToID100k)), dtype=np.int8)
    newUsers100k = sp.lil_matrix((len(newUserTrack100kStructure),len(newUserTracksIndexToID100k)), dtype=np.int8)

    
    for index, user in enumerate(userTrack100kStructure):
        trainTracks, testValTracks = train_test_split(userTrack100kStructure[user], test_size=0.30, random_state=42)
        valTracks, testTracks = train_test_split(testValTracks, test_size=0.5, random_state=42)
        #split the userTrack100kStructure into train, validation and test
        # 70% train, 15% validation, 15% test
        train100k[index, trainTracks] = 1
        validation100k[index, valTracks] = 1
        test100k[index, testTracks] = 1
        #     

    for index, user in enumerate(newUserTrack100kStructure):
        newUsers100k[index, newUserTrack100kStructure[user]] = 1

    # save the 100000x(number of songs per user * 100000) matrix
    sp.save_npz('../data/data100000/train100k.npz', train100k.tocsr())
    sp.save_npz('../data/data100000/validation100k.npz', validation100k.tocsr())
    sp.save_npz('../data/data100000/test100k.npz', test100k.tocsr())
    sp.save_npz('../data/data100000/newUsers100k.npz', newUsers100k.tocsr())

    # save the songIndexToID100k dictionary
    pickle.dump(songIndexToID100k, open('../data/data100000/songIndexToID100k.pkl', 'wb'))
    pickle.dump(newUserTracksIndexToID100k, open('../data/data100000/songIndexToID100knewUser.pkl', 'wb'))

    return 'Done'

print(get100000Users())