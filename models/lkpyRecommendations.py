import sys
import os
import pickle
import pandas as pd

# Load data
# Data manipulation
#subsetSize = int(sys.argv[1])
ratings = pd.read_csv('recBole/datasets/train100k.csv')

#Load model
model = sys.argv[1]
modelName = model + '100k'
print(modelName)
model = pickle.load(open(f'LKMFModels/{modelName}.pkl', 'rb'))
print(model)
numberOfRecommendations = int(sys.argv[2])


# Run through all users to get recommendations
recommendations = {}
for user in ratings['user'].unique(): 
    user_recs = model.recommend(user, numberOfRecommendations)
    recommendations[user] = user_recs

RecsImpMF = pd.concat(recommendations,keys=recommendations.keys())
RecsImpMF = RecsImpMF.reset_index(level=1, drop=True).reset_index()
RecsImpMF.rename(columns={'index':'user'}, inplace=True)
RecsImpMF.to_csv(f'LKMFRecs/{modelName}.csv', index=False)
print(f"Saved recommendations to LKMFRecs/{modelName}.csv")