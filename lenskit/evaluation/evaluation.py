import pandas as pd
from lenskit import batch, topn, util
from lenskit.metrics import topn as topn_metrics
from lenskit.topn import RecListAnalysis
from evaluationfunctions import item_coverage, gini
import pickle
import numpy as np
import time
import random
import json


truth = pd.read_csv('path/to/test/data', sep='\t')
truth.rename(columns={'user_id:token':'user', 'item_id:token':'item', 'rating:token':'rating'}, inplace=True)

model_names = ['ImpMF100k', 'popular100k','SVDBiased100k', 'SVDfunk100k']
k = 10

for model_name in model_names:
    model = pickle.load(open(f'path/to/model', 'rb'))
    
    #run subset of users
    #num_groups = 100
    #unique_users = truth['user'].unique()
    #user_groups = np.array_split(unique_users, num_groups)

    recommendations = batch.recommend(model, truth['user'].unique(), k, nprocs=1)

    #testframe = recommendations[recommendations['user'].isin(user_groups[0])]

    rla = RecListAnalysis()
    rla.add_metric(topn_metrics.recall, name='recall_10', k=10)
    rla.add_metric(topn_metrics.precision, name='precision_10', k=10)
    rla.add_metric(topn_metrics.hit, name='hit_10', k=10)
    rla.add_metric(topn_metrics.ndcg, name='ndcg_10', k=10)
    results = rla.compute(recommendations, truth)

    # Drop the 'nrecs' column
    results = results.drop(columns=['nrecs'])

    # Calculate the mean of each column
    average_metrics = results.mean()

    result_dict = average_metrics.to_dict()

    result_dict['giniindex'] = gini(recommendations)
    result_dict['itemCoverage'] = item_coverage(recommendations)


    json.dump(result_dict, open(f'path/to/evaluation/folder','w'))



