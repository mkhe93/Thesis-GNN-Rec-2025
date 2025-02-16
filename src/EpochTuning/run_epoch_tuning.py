import argparse
import threadpoolctl
import numpy as np
import torch
from tqdm import tqdm
from recbole_gnn.quick_start import run_recbole_gnn
from src.config import DATA_DIRECTORY, CONFIG_DIRECTORY

"""
    Output of these run should be the log files created by RecBole in:
        log -> <ModelName> -> <ModelName><DatasetName><Timestamp>.log
        
    used datasets for this search: [  9  25  68 104  88  80 139  95  99  54]
"""

if __name__ == "__main__":

    np.random.seed(100)
    random_numbers = np.random.randint(1, 177, size=10)

    # (Recbole identifier, config_file identifier)
    model_list = [('LightGCN', 'lightgcn'),
                  ('UltraGCN', 'ultragcn'),
                  ('SGL', 'sgl'),
                  ('XSimGCL', 'xsimgcl'),
                  ('ALS', 'als'),
                  ('BPR', 'bpr')
                  ]

    for model in model_list:

        test_res_list = []
        test_res_best_user_list = []
        test_res_worst_user_list = []

        for i in tqdm(random_numbers, desc="Datasets", unit='datasets'):

            torch.set_num_threads(8)

            config_dict = {
                'data_path': DATA_DIRECTORY.joinpath("mids-splits")
            }
            config_file = str(CONFIG_DIRECTORY.joinpath(f"{model[1]}.yaml"))

            parser = argparse.ArgumentParser()
            parser.add_argument("--config_files", type=str, default=config_file, help="config files")
            parser.add_argument('--dataset', '-d', type=str, default=f"mids-100000-{i}", help='name of datasets')
            parser.add_argument('--model', '-m', type=str, default=model[0], help='name of models')
            args, _ = parser.parse_known_args()

            # configurations initialization
            config_file_list = args.config_files.strip().split(' ') if args.config_files else None
            result = run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)