import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.config import PLOTS_DIRECTORY, DATA_DIRECTORY, ASSET_DIRECTORY, CONFIG_DIRECTORY, Colors
import os
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset
from recbole_gnn.data.dataset_metrics import GraphDatasetEvaluator
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gaussian_kde


"""
    Create data splits
"""

def load_data(filename: str, rows: int = None) -> pd.DataFrame:
    """
    Reads the filename as a pandas dataframe.

    Attributes:
    ----------
    filename: str
        the filename to read
    rows: int
        the number of entries to read from the file (default is `None` and reads the entire file)

    Returns:
    ----------
    df: pd.DataFRame
        pandas dataframe with the user-item interactions
    """
    with open(filename, 'r', encoding="utf-16") as f:
        objects = csv.reader(f, delimiter="\t")

        if rows is None:
            columns_list = list(objects)
        else:
            columns_list = [next(objects) for i in range(rows + 1)]

        df = pd.DataFrame(columns_list[1:], columns=columns_list[0])
        df = df.iloc[:, :3]
        if any(df.columns != ["userID", "itemID", "timestamp"]):
            df.columns = ['userID', 'itemID', 'timestamp']

        df['userID'] = df['userID'].astype(np.int64)
        df['itemID'] = df['itemID'].astype(np.int64)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.extract(r"^([\d-]+\s[\d:]+)")[0],
                                         format="%Y-%m-%d %H:%M:%S")

        df["timestamp"] = df["timestamp"].apply(lambda x: x.timestamp())

    return df

def plot_long_tail(n_downloads_per_user, n_downloads_per_track, save_fig: bool=False) -> None:
    # Function to compute the cumulative distribution
    def compute_cumulative_distribution(downloads):
        sorted_downloads = np.sort(downloads)[::-1]
        sorted_downloads = np.cumsum(sorted_downloads)
        cumulative_counts = np.arange(1, len(sorted_downloads) + 1)

        percentage_items = sorted_downloads / sorted_downloads.max() * 100
        cumulative_counts = cumulative_counts / len(cumulative_counts) * 100

        return percentage_items, cumulative_counts

    # Compute cumulative distributions
    x, y = compute_cumulative_distribution(n_downloads_per_track)

    plt.figure(figsize=(10, 7))
    plt.plot(x, y, label='items', color='red', linewidth=2, linestyle="--")
    plt.axvline(x=20, color='black', linestyle='--', linewidth=1)

    x, y = compute_cumulative_distribution(n_downloads_per_user)
    plt.plot(x, y, label='users', color='blue', linewidth=2)

    # Add text annotations
    plt.text(10, 10, 'Short-head\n(popular)', fontsize=18, ha='center')
    plt.text(70, 2, 'Long-tail\n(unpopular)', fontsize=18, ha='center')

    # Set logarithmic scale for y-axis
    plt.yscale('log')
    plt.ylim(0.1, 100)

    # Set labels and title
    plt.xlabel('% of downloads', fontsize=18)
    plt.ylabel('% of items', fontsize=18)
    # plt.title('Distribution of Items by Ratings', fontsize=16)
    plt.yticks([0.1, 1, 10, 100], ['0.1%', '1%', '10%', '100%'])
    plt.xticks([0, 20, 40, 60, 80, 100], ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlim(0, 100)

    plt.legend(loc='best', fontsize=18)

    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_edgecolor('black')    # Set the color of the border
        spine.set_linewidth(1)          # Set the thickness of the border

    plt.grid(True, which='both', linestyle=':', linewidth=0.5)

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath("dataset/long_tail_distribution.png"), dpi=300, transparent=True)

    plt.show()

def create_dataset_splits(dataset: pd.DataFrame, chunk_size: int=100000) -> None:
    # Directory path
    base_dir = DATA_DIRECTORY.joinpath("mids-splits")

    num_splits = dataset.shape[0] // chunk_size

    # Create and export each chunk
    for i, start_row in tqdm(enumerate(range(0, len(dataset), chunk_size), start=1), total=num_splits, unit='splits'):
        # Get the chunk
        chunk = dataset.iloc[start_row:start_row + chunk_size].copy()

        # Skip the last chunk if it has fewer rows than the chunk size
        if len(chunk) < chunk_size:
            break

        # Reset user and item IDs
        user_mapping = {user_id: new_id for new_id, user_id in enumerate(chunk['userID'].unique())}
        item_mapping = {item_id: new_id for new_id, item_id in enumerate(chunk['itemID'].unique())}
        chunk.loc[:, 'userID_reset'] = chunk['userID'].map(user_mapping)
        chunk.loc[:, 'itemID_reset'] = chunk['itemID'].map(item_mapping)

        # Rename columns
        chunk = chunk.rename(columns={
            'userID_reset': 'user_id:token',
            'itemID_reset': 'item_id:token',
            'userID': 'userID:token',
            'itemID': 'itemID:token',
            'timestamp': 'timestamp:float'
        })

        # Create directory for this chunk
        split_dir = os.path.join(base_dir, f'mids-100000-{i}')
        os.makedirs(split_dir, exist_ok=True)

        # File path
        file_path = os.path.join(split_dir, f'mids-100000-{i}.inter')

        # Save the chunk as a tab-separated file
        chunk.to_csv(file_path, sep='\t', index=False)

"""
    Calculcate characteristics
"""

def load_dataset_characteristics(file_path: str) -> pd.DataFrame:

    if os.path.isfile(file_path):
        print(f"File exists! Load file from {file_path}")
        df = pd.read_csv(file_path, sep='\t')

    else:
        print("File does not exist, calculate and save all dataset characteristics for each dataset..")

        dataset_eval_list = []
        for i in tqdm(range(1,177)):
            file_path = DATA_DIRECTORY.joinpath(f"mids-splits/mids-100000-{i}/mids-100000-{i}.inter")
            if not file_path.exists():
                break

            config_dict = {
                'data_path': DATA_DIRECTORY.joinpath("mids-splits")
            }
            config_file = CONFIG_DIRECTORY.joinpath("datasets.yaml")
            config = Config(model="BPR", dataset=f"mids-100000-{i}", config_file_list=[config_file],
                            config_dict=config_dict)

            # dataset filtering
            dataset = create_dataset(config)

            # calculate dataset metrics
            dataset_evaluator = GraphDatasetEvaluator(config, dataset)
            dataset_eval_dict = {"dataset": i}
            dataset_eval_dict.update(dataset_evaluator.evaluate())

            dataset_eval_list.append(dataset_eval_dict)

        df = pd.DataFrame(dataset_eval_list)
        df.to_csv(file_path, sep='\t', index=False)

    return df

def plot_column_distribution_seaborn(df: pd.DataFrame, column: str, value_name: str, show_fig=True, save_fig=False):
    """
    Plots the histogram of a column using Seaborn.

    Args:
        df (pd.DataFrame): The DataFrame containing dataset characteristics.
        column (str): The column to plot.
        dataset_range (tuple): Optional range for filtering datasets (lower_bound, upper_bound).
        value_name (str): Optional label for the x-axis.
        title_hist (str): Optional title for the histogram.
        save_fig (bool): Whether to save the figure.
    """
    # Prepare data
    values = df[column]

    if value_name is None:
        value_name = column

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.histplot(values, bins=30, stat='count', kde=False, color=Colors.TUD_BLUE, alpha=0.8)

    # Customize title and labels
    plt.xlabel(value_name, fontsize=25, labelpad=10)
    plt.ylabel("Frequency", fontsize=25, labelpad=10)

    # Customize ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save figure if required
    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath(f"dataset/dataset_hist_{column}.png"), dpi=300, transparent=True)

    if show_fig:
        plt.show()
    else:
        plt.close()

def plot_correlation_matrix(df: pd.DataFrame, columns: list, labels: list, filename: str="heatmap.png", save_fig: bool=False):

    correlation_data = pd.DataFrame()

    for col in columns:
        correlation_data[col] = df[col]

    # Compute the correlation matrix
    correlation_matrix = correlation_data.corr()

    # Create a list of colors
    colors = [Colors.TUD_BLUE, Colors.BU_GREEN1, Colors.ING_BLUE1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, yticklabels=labels, annot=True, cmap=cmap, center=0, fmt='.3f', linewidths=0.5)

    new_labels = labels

    # Get current x-tick positions and set the labels
    tick_positions = np.arange(len(correlation_matrix.columns)) + 0.4
    # Rotate y-tick labels
    plt.yticks(rotation=0, ha='right', va='center', fontsize=16)
    plt.xticks(ticks=tick_positions, labels=new_labels, rotation=45, ha='left', va='bottom', fontsize=16)
    plt.tick_params(axis='both', labelsize=16, labelbottom = False, bottom=False, top = False, labeltop=True)

    plt.gca().tick_params(axis='x')
    plt.gca().xaxis.set_label_position('top')

    plt.tight_layout()

    if save_fig:
        plt.savefig(ASSET_DIRECTORY.joinpath(f"plots/dataset/{filename}"), dpi=300, transparent=True)

    plt.show()