

import pickle


def createAlbumAndArtistMappings():

    # load the songToIndex for 100k and 100knewUsers
    songToIndex100k = pickle.load(open('../data/data100k/songIndexToID100k.pkl','rb'))
    songToIndex100knewUsers = pickle.load(open('../data/data100k/songIndexToID100knewUser.pkl','rb'))

    #load albumIdToIndex
    albumIdToIndex = pickle.load(open('../data/albumIdToIndex.p','rb'))

    #load artistIdToIndex
    artistIdToIndex = pickle.load(open('../data/artistIdToIndex.p','rb'))

    #create albumIdToIndex100k
    albumIdToIndex100k = {}
    for song in songToIndex100k.keys():
        albumId = songToIndex100k[song]['features']['albumId']
        if albumId not in albumIdToIndex100k.keys():
            albumIdToIndex100k[albumId] = len(albumIdToIndex100k)

    #create artistIdToIndex100k
    artistIdToIndex100k = {}
    for song in songToIndex100k.keys():
        artistId = songToIndex100k[song]['features']['artistId']
        if artistId not in artistIdToIndex100k.keys():
            artistIdToIndex100k[artistId] = len(artistIdToIndex100k)

    #create albumIdToIndex100knewUsers
    albumIdToIndex100knewUsers = {}
    for song in songToIndex100knewUsers.keys():
        albumId = songToIndex100knewUsers[song]['features']['albumId']
        if albumId not in albumIdToIndex100knewUsers.keys():
            albumIdToIndex100knewUsers[albumId] = len(albumIdToIndex100knewUsers)
    
    #create artistIdToIndex100knewUsers
    artistIdToIndex100knewUsers = {}
    for song in songToIndex100knewUsers.keys():
        artistId = songToIndex100knewUsers[song]['features']['artistId']
        if artistId not in artistIdToIndex100knewUsers.keys():
            artistIdToIndex100knewUsers[artistId] = len(artistIdToIndex100knewUsers)
    
    #save albumIdToIndex100k
    pickle.dump(albumIdToIndex100k, open('../data/data100k/albumIdToIndex100k.p','wb'))

    #save artistIdToIndex100k
    pickle.dump(artistIdToIndex100k, open('../data/data100k/artistIdToIndex100k.p','wb'))

    #save albumIdToIndex100knewUsers
    pickle.dump(albumIdToIndex100knewUsers, open('../data/data100k/albumIdToIndex100knewUsers.p','wb'))

    #save artistIdToIndex100knewUsers
    pickle.dump(artistIdToIndex100knewUsers, open('../data/data100k/artistIdToIndex100knewUsers.p','wb'))


    return print('done')

createAlbumAndArtistMappings()