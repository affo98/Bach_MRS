# This file contains the functions used to create the visualisations for the bachelor thesis.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import json
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# This function creates a nice scatter plot of the artist heterogeneity vs the length of the playlist.
# The data is taken from the AHdata100k.csv file that is created in evaluationfunctions.py
def artist_heterogeneity_scatter():
    inter = pd.read_csv('evaluation/AHdata100k.csv')
    inter.rename(columns={'user_id:token':'user', 'item_id:token':'item', 'rating:token':'rating'}, inplace=True)
    inter['artist_heterogeneity'] = inter['artist_heterogeneity'].astype(float)
    # the length of the playlist is grouped by user 
    playlist_length = inter.groupby('user').apply(lambda x: len(x['item'].unique())).reset_index(name='playlist_length')
    # now the dataframe contains a column with playlist length and artist heterogeneity for each user, which is then plotted
    inter = pd.merge(inter, playlist_length, on='user')
    sns.set_style("whitegrid")
    ax = sns.scatterplot(x="playlist_length", y="artist_heterogeneity", data=inter, alpha=0.5)
    ax.set(xlabel='Playlist length', ylabel='Artist heterogeneity')
    plt.savefig('../visualisations/artist_heterogeneity_scatter.png', dpi=300)
    return

# This functions creates a lineplot of the different models performance one metric at a time (precision, recall, ndcg, hitrate, gini_index, item_coverage)
# The data is taken from the folder ..recbole/recs file that is created in evaluationfunctions.py
    