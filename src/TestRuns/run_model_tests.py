# @Time   : 2023/2/13
# @Author : Gaowei Zhang
# @Email  : zgw2022101006@ruc.edu.cn


import argparse
import pandas as pd
from pathlib import Path
import torch

from tqdm import tqdm
from recbole_gnn.quick_start import run_recbole_gnn
from src.config import DATA_DIRECTORY, CONFIG_DIRECTORY, TESTRUN_DIRECTORY

if __name__ == "__main__":

    EVAL_RUN = "RO"
    # (Recbole identifier, config_file identifier)
    model_list = [('LightGCN', 'lightgcn'),
                  ('UltraGCN', 'ultragcn'),
                  ('AsymKNN', 'user_asym'),
                  ('AsymKNN', 'item_asym'),
                  ('SGL', 'sgl'),
                  ('XSimGCL', 'xsimgcl'),
                  ('ALS', 'als'),
                  ('BPR', 'bpr'),
                  ('MostPop', 'Pop')
                  ]

    for model in model_list:

        test_res_list = []
        test_res_best_user_list = []
        test_res_worst_user_list = []

        for i in tqdm(range(1,177)):

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

            # calculate dataset metrics
            test_res_dict = {"Model": model[0], "dataset": f"mids-100000-{i}"}
            test_res_best_user_dict = {"Model": model[0], "dataset": f"mids-100000-{i}"}
            test_res_worst_user_dict = {"Model": model[0], "dataset": f"mids-100000-{i}"}

            test_res_dict.update(result["test_result"])
            test_res_best_user_dict.update(result["best_user_evaluation"])
            test_res_worst_user_dict.update(result["worst_user_evaluation"])

            test_res_list.append(test_res_dict)
            test_res_best_user_list.append(test_res_best_user_dict)
            test_res_worst_user_list.append(test_res_worst_user_dict)

            # Combine all dictionaries into a single list
            combined_results = []

            for main_dict, best_dict, worst_dict in zip(test_res_list, test_res_best_user_list, test_res_worst_user_list):
                # Combine dictionaries with the desired prefixes
                combined_entry = {}
                combined_entry.update(main_dict)  # No prefix for the main dictionary
                combined_entry.update(
                    {f"best_user_{key}": value for key, value in best_dict.items() if key not in main_dict})
                combined_entry.update(
                    {f"worst_user_{key}": value for key, value in worst_dict.items() if key not in main_dict})

                # Append the combined entry to the results
                combined_results.append(combined_entry)

            # Convert the combined results into a DataFrame (optional)
            df = pd.DataFrame(combined_results)

            if model[0] == "AsymKNN":
                df.to_csv(TESTRUN_DIRECTORY.joinpath(f'results/LO/{model[1]}-{EVAL_RUN}.csv'), sep='\t', index=False)
            else:
                df.to_csv(TESTRUN_DIRECTORY.joinpath(f'results/LO/{model[0]}-{EVAL_RUN}.csv'), sep='\t', index=False)