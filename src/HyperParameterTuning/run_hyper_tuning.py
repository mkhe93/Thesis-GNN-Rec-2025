import argparse
import torch
from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function
from src.config import DATA_DIRECTORY, HYPER_DIRECTORY

# choose model out of this list: ["asymitemknn", "asymuserknn","als", "bpr", "lightgcn", "ultragcn", "sgl", "xsimgcl"]
model = "xsimgcl"

def main():

    config_dict = {
        'choice': {
            'dataset': ["mids-100000-1"],
            'data_path': [str(DATA_DIRECTORY.joinpath("mids-splits"))],
        }
    }

    with open(HYPER_DIRECTORY.joinpath(f"parameter_space/params_{model}.hyper"), "r") as file:
        for line in file:
            para_list = line.strip().split(" ")
            if len(para_list) < 3:
                continue
            config_dict["choice"].update({para_list[0]: eval("".join(para_list[2:]))})

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=str(HYPER_DIRECTORY.joinpath(f"parameter_fixed/hyper_{model}_djc100000.yaml")), help='fixed config files')
    parser.add_argument('--output_file', type=str, default=str(HYPER_DIRECTORY.joinpath(f"results/{model}-hyper-RO.result")), help='output file')
    args, _ = parser.parse_known_args()

    # set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set. Others: 'bayes','random'
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    hp = HyperTuning(objective_function, algo='bayes', max_evals=200, early_stop=50,
                     params_dict=config_dict, fixed_config_file_list=config_file_list)

    torch.set_num_threads(8)
    hp.run()
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

if __name__ == '__main__':
    main()