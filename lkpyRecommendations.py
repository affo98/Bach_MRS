import sys
import os
import pickle
import pandas as pd

# Load data
# Data manipulation
subsetSize = int(sys.argv[1])
ratings = pd.read_csv('dataSubsets/AllData'+str(subsetSize)+'.csv')

#Load model
modelPath = sys.argv[2]
modelName = modelPath.split('/')[-1].split('.')[0]
model = pickle.load(open(modelPath, 'rb'))

numberOfRecommendations = int(sys.argv[3])


# Run through all users to get recommendations
recommendations = {}
for user in ratings['user'].unique(): 
    user_recs = model.recommend(user, numberOfRecommendations)
    recommendations[user] = user_recs

RecsImpMF = pd.concat(recommendations,keys=recommendations.keys())
RecsImpMF = RecsImpMF.reset_index(level=1, drop=True).reset_index()
RecsImpMF.rename(columns={'index':'user'}, inplace=True)
RecsImpMF.to_csv(f'LKMFRecs/{modelName}'+str(subsetSize)+'.csv', index=False)