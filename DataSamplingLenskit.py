## A script the samples the data either from All, Train, Validation or test dependent algorithm
import scipy.sparse as sp
import pandas as pd
import numpy as np
import os


def sample(sample_size, sample_type:str):
    assert sample_type in ['all', 'train/test'], "sample_type must be 'a' or 'b'"

    """This function restructures the data into a pandas dataframe 
    where only interaction between users and items are stored in the form
    of a dataframe with columns [user,item,rating], where all ratings are 1."""
    subset = 1000000-sample_size
    if sample_type == 'train/test':
        ratingTrain = sp.load_npz('TrainUserItemMatrix.npz')
        ratingValid = sp.load_npz('ValidationUserItemMatrix.npz')
        ratingTest = sp.load_npz('TestUserItemMatrix.npz')
        listOfListsTrain = []
        for i in range(ratingTrain.shape[0]-subset):
            for j in ratingTrain[i].nonzero()[1].tolist():
                listOfListsTrain.append([i, j])
        train = pd.DataFrame(listOfListsTrain, columns=['user', 'item'])
        train['rating'] = 1

        listOfListsValidation = []
        for i in range(ratingValid.shape[0]-subset):
            for j in ratingTest[i].nonzero()[1].tolist():
                listOfListsValidation.append([i, j])
        validation = pd.DataFrame(listOfListsValidation, columns=['user', 'item'])
        validation['rating'] = 1

        listOfListsTest = []
        for i in range(ratingTest.shape[0]-subset):
            for j in ratingTest[i].nonzero()[1].tolist():
                listOfListsTest.append([i, j])
        test = pd.DataFrame(listOfListsTest, columns=['user', 'item'])
        test['rating'] = 1

        return train, validation, test
    
    if sample_type == 'all':
        ratingAll = sp.load_npz('UserItemMatrix.npz')
        listOfListsAll = []
        for i in range(ratingAll.shape[0]-subset):
            for j in ratingAll[i].nonzero()[1].tolist():
                listOfListsAll.append([i, j])
        All = pd.DataFrame(listOfListsAll, columns=['user', 'item'])
        All['rating'] = 1
        return All
    
for subsetSize in [5000,10000,50000,100000,500000]:
    os.mkdir('dataSubsets')
    sample(subsetSize, 'all').to_csv('dataSubsets/AllData'+str(subsetSize)+'.csv', index=False)
    train, test, validation = sample(subsetSize, 'train/test')
    train.to_csv('dataSubsets/TrainData'+str(subsetSize)+'.csv', index=False)
    test.to_csv('dataSubsets/TestData'+str(subsetSize)+'.csv', index=False)
    validation.to_csv('dataSubsets/ValidationData'+str(subsetSize)+'.csv', index=False)