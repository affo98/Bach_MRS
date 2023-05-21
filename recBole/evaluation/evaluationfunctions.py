import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from tqdm import tqdm

def item_coverage(dataframe):
    recommended_items = len(dataframe['item'].unique())
    coverage = recommended_items / 7348433 # len of items in the dataset
    
    return coverage


def gini(df):
    array = df['item'].astype(float).to_numpy()
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    #array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    #print(array)
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def artist_heterogeneity():
    import pickle
    songs = pickle.load(open('path/to/songs_mapping_dictionary/for/dataset', 'rb'))
    train_inter = pd.read_csv('path/to/train/data', sep='\t')
    train_inter.rename(columns={'user_id:token':'user', 'item_id:token':'item', 'rating:token':'rating'}, inplace=True)
    #retrieve artist id for each song form the songs dict and add it to the train_inter dataframe
    for song_id in tqdm(train_inter['item'].unique()):
        artist_id = int(songs[song_id]['features']['artistId'])
        train_inter.loc[train_inter['item'] == song_id, 'artist'] = artist_id

    #calculate artist heterogeneity that is defined by the number of unique songs divided by the unique number of artists, grouped by user (in each playlist).
    artist_heterogenity = train_inter.groupby('user').apply(lambda x: log2(len(x['item'].unique()) / len(x['artist'].unique())))
    #add artist heterogeneity to the train_inter dataframe
    train_inter = train_inter.merge(artist_heterogenity, on='user')
    train_inter.rename(columns={0:'artist_heterogeneity'}, inplace=True)
    #store dataframe now with artist id and artist heterogeneity 
    train_inter.to_csv('/path/to/evaluation/AHdata100k.csv', index=False)
    return artist_heterogenity



AH = artist_heterogeneity()
print(AH)