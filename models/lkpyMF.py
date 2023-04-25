import sys

# Models
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.basic import Popular

# Item-Recommendation
from lenskit.algorithms import Recommender

# Dependencies
import pandas as pd 
import numpy as np
import scipy.sparse as sp
import pickle
from tqdm import tqdm

subsetSize = int(sys.argv[1])
numIterations = int(sys.argv[2])
numFeatures = int(sys.argv[3])


## Define the models
ImpMF = ImplicitMF(features=numFeatures, iterations=numIterations, method='lu')
ImpMF = Recommender.adapt(ImpMF)

SVDBiased = BiasedSVD(features=numFeatures)
SVDBiased = Recommender.adapt(SVDBiased)

SVDfunk = FunkSVD(features=numFeatures, iterations=numIterations)
SVDfunk = Recommender.adapt(SVDfunk)

popular = Popular()
popular = Recommender.adapt(popular)


## Load the data
ratings = pd.read_csv('dataSubsets/AllData'+str(subsetSize)+'.csv')

## Fit each of the models on the data and recommened 10 items to each user for each model
ImpMF.fit(ratings) # Fit The model
pickle.dump(ImpMF, open('LKMFModels/ImpMF'+str(subsetSize)+'.pkl','wb')) # Save the model

SVDBiased.fit(ratings)
pickle.dump(SVDBiased, open('LKMFModels/SVDBiased'+str(subsetSize)+'.pkl','wb'))

SVDfunk.fit(ratings)
pickle.dump(SVDfunk, open('LKMFModels/SVDfunk'+str(subsetSize)+'.pkl','wb'))

popular.fit(ratings)
pickle.dump(popular, open('LKMFModels/popular'+str(subsetSize)+'.pkl','wb'))

