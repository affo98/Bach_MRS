

import pandas as pd
import scipy.sparse as sp
import numpy as np
import pickle

def convertsp_csv(input_file,output_file):
    ratings = sp.load_npz(input_file)
    listOfLists = []
    for i in range(ratings.shape[0]):
        for j in ratings[i].nonzero()[1].tolist():
            listOfLists.append([i, j])
    df = pd.DataFrame(listOfLists, columns=['user', 'item'])
    df['rating'] = 1
    return df.to_csv(output_file, index=False)


def convertnp_csv(input_file, output_file):
    songToIndex = pickle.load(open(input_file, 'rb'))

    temporaryLL = [list(songToIndex[i]['features'].values()) for i in songToIndex]


    df = pd.DataFrame(temporaryLL,columns=songToIndex[0]['features'].keys())
    print(df.head())
    #drop album and artist id's to get only song features
    df = df.drop(columns=['albumId', 'artistId'])
    print(df.head())
    return df.to_csv(output_file, index=False)

input_files = ['../data/data100k/train100k.npz', '../data/data100k/test100k.npz', '../data/data100k/validation100k.npz', '../data/data100k/newUsers100k.npz']
output_files = ['../recBole/train100k.csv', '../recBole/test100k.csv', '../recBole/validation100k.csv', '../recBole/newUsers100k.csv']

#for input_file, output_file in zip(input_files, output_files):
   # convertsp_csv(input_file, output_file)

input_dicts = ['../data/data100k/songIndexToID100k.pkl', '../data/data100k/songIndexToID100knewUser.pkl']
output_dicts = ['../recBole/items100k.csv', '../recBole/itemsNewUsers.csv']

for input_dict, output_dict in zip(input_dicts, output_dicts):
    convertnp_csv(input_dict, output_dict)