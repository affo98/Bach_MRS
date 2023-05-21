import torch
import recbole
import json
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NeuMF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.trainer.hyper_tuning import HyperTuning
from recbole.quick_start import objective_function


if __name__ == '__main__':

    # paramater configurations initialization
    config_path = 'NeuMF.yaml'  # set your desired config file path
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

    # # model loading
    model = NeuMF(config, train_data.dataset).to(device)
    logger.info(model)

    # # trainer loading and initialization
    trainer = Trainer(config, model)

    # # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    
    epochs = config['epochs']
    # # save the trained model
    model_path = f'models/NeuMF{epochs}.pt'  # set your desired file path and name
    torch.save(model.state_dict(), model_path)
    
    # model evaluation
    test_result = trainer.evaluate(test_data)
    json.dump(test_result, open(f'../results/NeuMF{epochs}.json', 'w'))