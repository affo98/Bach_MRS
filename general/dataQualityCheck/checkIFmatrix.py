import numpy as np
import json

#check if there is actually any data in the itemFeatures.npy file
# Check in last (progress - 1000) to index in progress.json   

itemFeatures = np.load('data/itemFeatures.npy')

with open('progress/progress.json', 'r') as f:
    progress = int(json.load(f))


#[0, 0, 70, 4, 0.904, 0.813, 4, -7.105, 0, 0.121, 0.0311, 0.00697, 0.0471, 0.81, 125.461, 4]
# sample 10 different rows from the itemFeatures.npy file
for i in range(100):
    if np.sum(itemFeatures[np.random.randint(progress-10000, progress)]) == 0:
        raise ValueError('There is no data in the itemFeatures.npy file')
