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


#subsetSize = int(sys.argv[1])
#numIterations = int(sys.argv[2])
#numFeatures = int(sys.argv[3])


## Define the models
ImpMF = ImplicitMF(features=32, iterations=25, method='lu', reg=125, weight=30,progress=tqdm)
ImpMF = Recommender.adapt(ImpMF)

# #n_iterations > 5 is considered large number of iterations 
SVDBiased = BiasedSVD(features=32, damping=8)
SVDBiased = Recommender.adapt(SVDBiased)

SVDfunk = FunkSVD(features=150, iterations=100)
SVDfunk = Recommender.adapt(SVDfunk)

popular = Popular()
popular = Recommender.adapt(popular)


## Load the data
ratings = pd.read_csv('path/to/train/data', sep='\t')

ratings.rename(columns={'user_id:token':'user', 'item_id:token':'item', 'rating:token':'rating'}, inplace=True)
## Fit each of the models on the data and recommened 10 items to each user for each model
ImpMF.fit(ratings) # Fit The model
pickle.dump(ImpMF, open('path/to/models/optImpMF100k.pkl','wb')) # Save the model

SVDBiased.fit(ratings)
pickle.dump(SVDBiased, open('models/optSVDBiased100k.pkl','wb'))

SVDfunk.fit(ratings)
pickle.dump(SVDfunk, open('../LKMFModels/SVDfunk100k.pkl','wb'))

popular.fit(ratings)
pickle.dump(popular, open('../LKMFModels/popular100k.pkl','wb'))

