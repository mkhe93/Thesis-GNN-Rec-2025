from recbole_gnn.quick_start import run_recbole_gnn
import torch
from src.config import DATA_DIRECTORY, CONFIG_DIRECTORY, PROJECT_DIRECTORY

NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)

if __name__ == '__main__':

    model = 'UltraGCN'
    config_files = str(CONFIG_DIRECTORY.joinpath('ultragcn.yaml'))
    dataset = 'mids-100000'
    config_dict = {
        'data_path': DATA_DIRECTORY.joinpath(''),
    }

    run_recbole_gnn(model=model, dataset=dataset, config_file_list=[config_files], config_dict=config_dict, saved=False)