import torch
import numpy as np
import pickle
import pandas as pd
import recbole
import json
from recbole.quick_start import load_data_and_model
from recbole.data import create_dataset
from recbole.config import Config
from recbole.utils import init_logger, get_local_time 
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.bpr import BPR
from recbole.trainer import Trainer
from evaluationfunctions import gini, item_coverage
from tqdm import tqdm
print('RecBole Version: ', recbole.__version__)
print('Torch Version: ', torch.__version__)

def get_topk_recommendations(u,k):

    # # Get Test Set Out

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
         model_file='path/to/model')
    
    # convert user_ids to strings to parse into dataset.token2id

    user_ids = [str(i) for i in range(0,dataset.user_num-1)]

    
    uid_series = dataset.token2id(dataset.uid_field, user_ids)

    #Load Trained model:
    config = Config(config_file_list=['BPR.yaml'])

    model = BPR(config, dataset).to('cpu')

    model.load_state_dict(torch.load('models/BPR100.pt', map_location='cpu'))

    topk_scores, topk_iid_list = full_sort_topk(uid_series, model, test_data,k=k, device='cpu')

    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    data = []
    for user_id, recommendations in enumerate(external_item_list):

        for rank, item in enumerate(recommendations):
            itemShowcase = [None] * 3
            itemShowcase[0] = user_id
            itemShowcase[1] = item
            itemShowcase[2] = rank+1
            data.append(itemShowcase)

    df = pd.DataFrame(data, columns=['user','item','rank'])

    test_result = {'gini': gini(df), 'item_coverage': item_coverage(df)}

    json.dump(test_result, open(f'../evaluation/model/name/gini/itemCoverage','w'))

    df.to_csv(f'../recs/recommendations/model/name', index=False)

    return 

recommendations = get_topk_recommendations(100,10)

