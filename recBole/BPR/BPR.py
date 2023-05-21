import torch
import json
from logging import getLogger
import logging
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.utils import get_local_time, ensure_dir
from recbole.trainer.hyper_tuning import HyperTuning
from recbole.quick_start import objective_function

if __name__ == '__main__':

    # Initialize multiprocessing for PyTorch
    #mp.set_start_method('fork', force=True)
    # paramater configurations initialization
    config_path = 'BPR.yaml'  # set your desired config file path
    config = Config(config_file_list=[config_path])
    
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    device = torch.device(config['device'])

    # check if colab is using GPU
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        print("Using GPU")

    hyper_tuning = HyperTuning(
        objective_function=objective_function,
        params_file='hyper.test',
        fixed_config_file_list=[config_path],
    )

    best_params = hyper_tuning.run()

    config['hyper_parameters'].update(best_params)

    # # model loading and initialization
    model = BPR(config, train_data.dataset).to(config['device'])

    logger.info(model)

    # # trainer loading and initialization
    trainer = Trainer(config, model)

    # # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    
    epochs = config['train_epoch']
    # # save the trained model
    model_path = f'models/BPR{epochs}.pt'  # set your desired file path and name
    torch.save(model.state_dict(), model_path)
    
    # model evaluation
    test_result = trainer.evaluate(test_data)
    json.dump(test_result, open(f'../results/BPR{epochs}.json', 'w'))