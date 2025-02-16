import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from scipy.stats import probplot
import pandas as pd
import numpy as np
import ast
import json
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset
from recbole_gnn.data.dataset_metrics import GraphDatasetEvaluator
from tqdm import tqdm
import os

from src.config import (Colors, process_user_columns,
                        PROJECT_DIRECTORY, EVALUATION_DIRECTORY, TESTRUN_DIRECTORY,
                        DATA_DIRECTORY, CONFIG_DIRECTORY, PLOTS_DIRECTORY,
                        ASSET_DIRECTORY, json_loads_user_columns)


"""
    Create Evaluation Assets
"""

def load_benchmark_datasets(eval_run: str="RO") -> pd.DataFrame:

    csv_files = list(TESTRUN_DIRECTORY.glob(f"results/{eval_run}/*.csv"))
    dfs = [pd.read_csv(file, sep="\t") for file in csv_files]

    df_combined = pd.concat(dfs, ignore_index=True)

    df_combined['dataset'] = df_combined['dataset'].str.extract(r'-(\d+)$').astype(int)
    df_combined["eval_type"] = [eval_run] * len(df_combined)
    df_combined = process_user_columns(df_combined)

    return df_combined

def get_users_topological_chars(data: pd.DataFrame, file_path: str, num_datasets: int=176) -> pd.DataFrame:

    if os.path.isfile(file_path):
        print(f"File exists! Load file from {file_path}")
        df = pd.read_csv(file_path, sep='\t')
        df = process_user_columns(df)

    else:
        print("File does not exist, calculate all user's topology characteristics for each dataset..")

        dataset_eval_list = []
        datasets = data['dataset'].head(num_datasets).unique()
        for i in tqdm(datasets, total=len(datasets), unit='datasets'):

            config_dict = {
                'data_path': DATA_DIRECTORY.joinpath("mids-splits")
            }
            config_file = CONFIG_DIRECTORY.joinpath("datasets.yaml")

            config = Config(model="BPR", dataset=f"mids-100000-{i}", config_file_list=[config_file],
                            config_dict=config_dict)
            dataset = create_dataset(config)

            # Extract unique user IDs from best_user_ and worst_user_ columns
            best_user_ids = set()
            worst_user_ids = set()
            for index, row in data[data['dataset'] == i].iterrows():
                for col in data.columns:
                    if col.startswith('best_user_'):
                        for entry in row[col]:
                            best_user_ids.add(int(list(entry.keys())[0]))
                    if col.startswith('worst_user_'):
                        for entry in row[col]:
                            worst_user_ids.add(int(list(entry.keys())[0]))


            # calculate dataset metrics
            dataset_evaluator = GraphDatasetEvaluator(config, dataset)
            dataset_eval_dict = {"dataset": i}
            dataset_eval_dict.update(dataset_evaluator.evaluate_best_worst_users(best_user_ids, worst_user_ids))
            dataset_eval_list.append(dataset_eval_dict)

            df = pd.DataFrame(dataset_eval_list)
            df.to_csv(file_path, sep='\t', index=False) # overwrite file after each iteration

    return df

def translate_userids(df: pd.DataFrame, num_rows: int = 10) -> pd.DataFrame:

    data = df.copy()
    # Extract columns ending with @[10]
    columns_to_process = [col for col in data.columns if col.endswith("@[10]")]
    for index, row in tqdm(data.iloc[:num_rows].iterrows(), total=num_rows, unit='rows'):
        # configurations initialization
        filename = DATA_DIRECTORY.joinpath(
            f"mids-splits/mids-100000-{row['dataset']}/mids-100000-{row['dataset']}.inter")
        db = pd.read_csv(filename, sep="\t", encoding="utf-8")

        config_dict = {
            'data_path': DATA_DIRECTORY.joinpath("mids-splits")
        }
        config_file = CONFIG_DIRECTORY.joinpath("datasets.yaml")
        config = Config(model="BPR", dataset=f"mids-100000-{row['dataset']}", config_file_list=[config_file],
                        config_dict=config_dict)
        dataset = create_dataset(config)

        # mapping recbole ID -> local ID (as recbole resets the index after filtering the datasets)
        flipped_dict = {v: k for k, v in dataset.field2token_id['user_id'].items()}

        # mapping local ID -> global ID
        translation_dict = dict(zip(db["user_id:token"], db["userID:token"]))

        # Process each column
        for col in columns_to_process:
            # Parse the string entries and extract the user IDs
            new_entry = {
                str(translation_dict[int(flipped_dict[int(list(entry.keys())[0])])]): float(list(entry.values())[0]) for
                entry in row[col]}
            data.at[index, col] = new_entry

    return data

def build_users_popularity_lookup_file(file_path: str, num_datasets: int = 10) -> pd.DataFrame:

    if os.path.isfile(file_path):
        print(f"File exists! Load file from {file_path}")
        # Load the existing DataFrame
        df = pd.read_csv(file_path, sep='\t')
        # Convert the user_popularity column back to dictionaries
        if 'user_popularity' in df.columns:
            df['user_popularity'] = df['user_popularity'].apply(ast.literal_eval)

    else:
        print("File does not exist, calculate all user's popularity for each dataset..")

        # Initialize DataFrame with pre-defined size and columns
        df = pd.DataFrame({'dataset': range(1, num_datasets + 1)})
        df['user_popularity'] = [{} for _ in range(num_datasets)]

        # Loop through each dataset and compute user popularity
        for i in tqdm(range(num_datasets)):
            filename = DATA_DIRECTORY.joinpath(
                f"mids-splits/mids-100000-{i}/mids-100000-{i}.inter")
            db = pd.read_csv(filename, sep="\t", encoding="utf-8")

            # Filter users with less than 20 interactions
            db['interaction_count'] = db.groupby("userID:token")["itemID:token"].transform('count')
            filtered_df = db[db['interaction_count'] >= 20].copy()

            # Calculate item popularity
            item_popularity = filtered_df.groupby("itemID:token")["userID:token"].nunique()

            # Map item popularity back to filtered_df safely
            filtered_df.loc[:, 'item_popularity'] = filtered_df["itemID:token"].map(item_popularity)

            # Compute user average popularity
            user_avg_popularity = filtered_df.groupby("userID:token")['item_popularity'].agg(['mean', 'median'])

            # Compute user popularity dictionary
            all_users_popularity_dict = {
                user_id: (row['mean'], row['median'])
                for user_id, row in user_avg_popularity.iterrows()
            }

            # Assign the dictionary to the DataFrame
            df.at[i, 'user_popularity'] = all_users_popularity_dict

        df.to_csv(file_path, sep='\t', index=False)

    return df

def get_user_popularity(data: pd.DataFrame, popularity_lookup: dict, num_rows: int=10) -> pd.DataFrame:

    df = data.copy()

    # Initialize new columns with empty lists
    df['best_users_mean_popularity_dict'] = [{} for _ in range(len(df))]
    df['best_users_mean_popularity_mean'] = [{} for _ in range(len(df))]
    df['best_users_mean_popularity_max'] = [{} for _ in range(len(df))]
    df['best_users_mean_popularity_min'] = [{} for _ in range(len(df))]
    df['best_users_median_popularity_dict'] = [{} for _ in range(len(df))]
    df['best_users_median_popularity_mean'] = [{} for _ in range(len(df))]
    df['best_users_median_popularity_max'] = [{} for _ in range(len(df))]
    df['best_users_median_popularity_min'] = [{} for _ in range(len(df))]
    df['best_users_node_degree_dict'] = [{} for _ in range(len(df))]
    df['best_users_node_degree_mean'] = [{} for _ in range(len(df))]
    df['best_users_node_degree_median'] = [{} for _ in range(len(df))]
    df['best_users_node_degree_min'] = [{} for _ in range(len(df))]
    df['best_users_node_degree_max'] = [{} for _ in range(len(df))]

    df['worst_users_mean_popularity_dict'] = [{} for _ in range(len(df))]
    df['worst_users_mean_popularity_mean'] = [{} for _ in range(len(df))]
    df['worst_users_mean_popularity_max'] = [{} for _ in range(len(df))]
    df['worst_users_mean_popularity_min'] = [{} for _ in range(len(df))]
    df['worst_users_median_popularity_dict'] = [{} for _ in range(len(df))]
    df['worst_users_median_popularity_mean'] = [{} for _ in range(len(df))]
    df['worst_users_median_popularity_max'] = [{} for _ in range(len(df))]
    df['worst_users_median_popularity_min'] = [{} for _ in range(len(df))]
    df['worst_users_node_degree_dict'] = [{} for _ in range(len(df))]
    df['worst_users_node_degree_mean'] = [{} for _ in range(len(df))]
    df['worst_users_node_degree_median'] = [{} for _ in range(len(df))]
    df['worst_users_node_degree_min'] = [{} for _ in range(len(df))]
    df['worst_users_node_degree_max'] = [{} for _ in range(len(df))]

    df['all_users_median_popularity_mean'] = [{} for _ in range(len(df))]
    df['all_users_median_popularity_max'] = [{} for _ in range(len(df))]
    df['all_users_median_popularity_min'] = [{} for _ in range(len(df))]
    df['all_users_node_degree_mean'] = [{} for _ in range(len(df))]
    df['all_users_median_node_degree_median'] = [{} for _ in range(len(df))]
    df['all_users_node_degree_max'] = [{} for _ in range(len(df))]
    df['all_users_node_degree_min'] = [{} for _ in range(len(df))]

    # Loop through each row
    for index, row in tqdm(df.iloc[:num_rows].iterrows(), total=num_rows, unit='rows'):

        # configurations initialization
        filename = DATA_DIRECTORY.joinpath(
            f"mids-splits/mids-100000-{row['dataset']}/mids-100000-{row['dataset']}.inter")
        db = pd.read_csv(filename, sep="\t", encoding="utf-8")

        # Filter users with less than 20 node_degree
        db['interaction_count'] = db.groupby("userID:token")["itemID:token"].transform('count')
        filtered_df = db[db['interaction_count'] >= 20].copy()
        all_users_node_degree_dict = filtered_df.groupby("userID:token")['interaction_count'].first()

        # Extract unique user IDs from best_user_ and worst_user_ columns
        best_user_ids = set()  # Use a set to ensure uniqueness
        worst_user_ids = set()  # Use a set to ensure uniqueness
        for col in df.columns:
            if col.startswith('best_user_'):
                # Extract user IDs from the list of dictionaries
                for entry in row[col].keys():
                    best_user_ids.add(int(entry))  # Add user ID (the key) to the set
            if col.startswith('worst_user_'):
                # Extract user IDs from the list of dictionaries
                for entry in row[col].keys():  # Assuming each entry is a list of dictionaries
                    worst_user_ids.add(int(entry))  # Add user ID (the key) to the set

        # Prepare best users
        best_users_mean_popularity_dict = {
            str(user_id): popularity_lookup.loc[row['dataset'] - 1, 'user_popularity'][user_id][0] for user_id
            in best_user_ids}
        best_users_median_popularity_dict = {
            str(user_id): popularity_lookup.loc[row['dataset'] - 1, 'user_popularity'][user_id][1] for user_id
            in best_user_ids}
        best_users_node_degree_dict = {str(user_id): float(all_users_node_degree_dict[user_id]) for user_id in
                                       best_user_ids}
        best_users_mean_values_list = list(best_users_mean_popularity_dict.values())
        best_users_median_values_list = list(best_users_median_popularity_dict.values())
        best_users_node_degree_list = list(best_users_node_degree_dict.values())

        worst_users_mean_popularity_dict = {
            str(user_id): popularity_lookup.loc[row['dataset'] - 1, 'user_popularity'][user_id][0] for user_id
            in worst_user_ids}
        worst_users_median_popularity_dict = {
            str(user_id): popularity_lookup.loc[row['dataset'] - 1, 'user_popularity'][user_id][1] for user_id
            in worst_user_ids}
        worst_users_node_degree_dict = {str(user_id): float(all_users_node_degree_dict[user_id]) for user_id in
                                        worst_user_ids}
        worst_users_mean_values_list = list(worst_users_mean_popularity_dict.values())
        worst_users_median_values_list = list(worst_users_median_popularity_dict.values())
        worst_users_node_degree_list = list(worst_users_node_degree_dict.values())

        all_users_mean_values_list = [entry[0] for entry in list(
            popularity_lookup.loc[row['dataset'] - 1, 'user_popularity'].values())]
        all_users_median_values_list = [entry[1] for entry in list(
            popularity_lookup.loc[row['dataset'] - 1, 'user_popularity'].values())]

        # Assign
        df.at[index, 'best_users_mean_popularity_dict'] = best_users_mean_popularity_dict
        df.at[index, 'best_users_mean_popularity_mean'] = np.mean(best_users_mean_values_list)
        df.at[index, 'best_users_mean_popularity_max'] = np.max(best_users_mean_values_list)
        df.at[index, 'best_users_mean_popularity_min'] = np.min(best_users_mean_values_list)
        df.at[index, 'best_users_median_popularity_dict'] = best_users_median_popularity_dict
        df.at[index, 'best_users_median_popularity_mean'] = np.mean(best_users_median_values_list)
        df.at[index, 'best_users_median_popularity_max'] = np.max(best_users_median_values_list)
        df.at[index, 'best_users_median_popularity_min'] = np.min(best_users_median_values_list)
        df.at[index, 'best_users_node_degree_dict'] = best_users_node_degree_dict
        df.at[index, 'best_users_node_degree_mean'] = np.mean(best_users_node_degree_list)
        df.at[index, 'best_users_node_degree_median'] = np.median(best_users_node_degree_list)
        df.at[index, 'best_users_node_degree_min'] = np.min(best_users_node_degree_list)
        df.at[index, 'best_users_node_degree_max'] = np.max(best_users_node_degree_list)

        df.at[index, 'worst_users_mean_popularity_dict'] = worst_users_mean_popularity_dict
        df.at[index, 'worst_users_mean_popularity_mean'] = np.mean(worst_users_mean_values_list)
        df.at[index, 'worst_users_mean_popularity_max'] = np.max(worst_users_mean_values_list)
        df.at[index, 'worst_users_mean_popularity_min'] = np.min(worst_users_mean_values_list)
        df.at[index, 'worst_users_median_popularity_dict'] = worst_users_median_popularity_dict
        df.at[index, 'worst_users_median_popularity_mean'] = np.mean(worst_users_median_values_list)
        df.at[index, 'worst_users_median_popularity_max'] = np.max(worst_users_median_values_list)
        df.at[index, 'worst_users_median_popularity_min'] = np.min(worst_users_median_values_list)
        df.at[index, 'worst_users_node_degree_dict'] = worst_users_node_degree_dict
        df.at[index, 'worst_users_node_degree_mean'] = np.mean(worst_users_node_degree_list)
        df.at[index, 'worst_users_node_degree_median'] = np.median(worst_users_node_degree_list)
        df.at[index, 'worst_users_node_degree_min'] = np.min(worst_users_node_degree_list)
        df.at[index, 'worst_users_node_degree_max'] = np.max(worst_users_node_degree_list)

        df.at[index, 'all_users_mean_popularity_mean'] = np.mean(all_users_mean_values_list)
        df.at[index, 'all_users_mean_popularity_max'] = np.max(all_users_mean_values_list)
        df.at[index, 'all_users_mean_popularity_min'] = np.min(all_users_mean_values_list)
        df.at[index, 'all_users_median_popularity_mean'] = np.mean(all_users_median_values_list)
        df.at[index, 'all_users_median_popularity_max'] = np.max(all_users_median_values_list)
        df.at[index, 'all_users_median_popularity_min'] = np.min(all_users_median_values_list)
        df.at[index, 'all_users_node_degree_mean'] = all_users_node_degree_dict.mean()
        df.at[index, 'all_users_median_node_degree_median'] = all_users_node_degree_dict.median()
        df.at[index, 'all_users_node_degree_max'] = all_users_node_degree_dict.max()
        df.at[index, 'all_users_node_degree_min'] = all_users_node_degree_dict.min()

    return df


"""
    Model Analysis
"""

def get_evaluation_data(eval_run: str="RO") -> pd.DataFrame:

    file_path = EVALUATION_DIRECTORY.joinpath(f"{eval_run}/EvaluationData.parquet")
    if not os.path.isfile(file_path):
        print("File does not exist, run the BuildEvalDataset.ipynb first")
    else:
        print(f"File exists! Load file from {file_path}")
        df = pd.read_parquet(file_path, engine="pyarrow")
        df = json_loads_user_columns(df)
        return df


def plot_metrics_by_dataset_with_boxplot(df: pd.DataFrame, models_dict: dict, eval_run: str, metric: str ="NDCG@10",
                                         save_fig: bool = False, with_scatter: bool =False) -> None:
    """
    Generates a boxplot to visualize the distribution of a specified metric across different models in a dataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing model performance data.
        models_dict (dict): A dictionary mapping model identifiers (keys) to display names (values).
        metric (str, optional): The name of the metric column to be visualized. Defaults to "NDCG@10".
        save_fig (bool, optional): If True, saves the figure as a PNG file. Defaults to False.
        with_scatter (bool, optional): If True, overlays a scatter plot on top of the boxplot for additional visualization. Defaults to False.

    Raises:
        ValueError: If the specified metric column is not found in the DataFrame.

    Returns:
        None: The function generates and displays a boxplot, optionally saving it to a file.
    """

    # Ensure the 'metric' column exists in the dataframe
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe columns.")

    models = list(models_dict.keys())
    model_names = list(models_dict.values())

    if models:
        df = df[df['Model'].isin(models)]
        # Map the 'Model' column to the new model names using a dictionary
        model_mapping = dict(zip(models, model_names))
        df.loc[:, 'Model'] = df['Model'].map(model_mapping)
        df.loc[:, 'Model'] = pd.Categorical(df['Model'], categories=model_names, ordered=True)

    # Create the plot
    plt.figure(figsize=(12, 6))
    boxplot = sns.boxplot(
        x='Model', y=metric, data=df, showmeans=True, meanline=True,
        meanprops={"linestyle": "--", "color": "black"},
        hue='Model', dodge=False, legend=False, boxprops=dict(alpha=.3)
    )
    if with_scatter:
        sns.stripplot(data=df, x='Model', y=metric, color=Colors.BU_GREEN3, alpha=0.2, jitter=True)

    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_edgecolor('black')    # Set the color of the border
        spine.set_linewidth(1)          # Set the thickness of the border

    # Customize plot
    plt.xlabel("")  # Turn off x-axis label
    plt.ylabel("$NDCG@10$", fontsize=25, labelpad=10)
    plt.xticks(rotation=0, fontsize=20)

    plt.yticks(fontsize=20)

    plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

    plt.tight_layout()

    if save_fig:
        if with_scatter:
            plt.savefig(PLOTS_DIRECTORY.joinpath(f"evaluation/boxplot_{metric}_scatter_{eval_run}.png"), dpi=300, transparent=True)
        else:
            plt.savefig(PLOTS_DIRECTORY.joinpath(f"evaluation/boxplot_{metric}_{eval_run}.png"), dpi=300, transparent=True)

    plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                            models: list, metric_labels: dict = None, metrics: list = None,
                            transformation: str = 'log', save_fig: bool = False, filename: str = "") -> None:
    """
    Generates and visualizes a correlation matrix heatmap for specified metrics across different models.

    Args:
        df (pd.DataFrame): The input DataFrame containing model performance data.
        models (list): A list of model names to filter and include in the analysis.
        metric_labels (dict, optional): A dictionary mapping metric names to display labels. Defaults to None.
        metrics (list, optional): A list of metrics to be included in the correlation matrix. Defaults to ['best_users_mean_popularity_mean'].
        transformation (str, optional): The data transformation method to apply ('log', 'sqrt', or None). Defaults to 'log'.
        save_fig (bool, optional): If True, saves the figure as a PNG file. Defaults to False.
        filename (str, optional): The filename for saving the figure. Must be provided if save_fig is True.

    Raises:
        ValueError: If save_fig is True but filename is not provided.

    Returns:
        None: The function generates and displays a correlation matrix heatmap, optionally saving it to a file.
    """

    if metrics is None:
        metrics = ['best_users_mean_popularity_mean']

    if save_fig and filename == "":
        raise ValueError(f"You need to define a proper filename to save the figure.")

    model_names = models

    if models:
        df = df[df['Model'].isin(models)]
        df.loc[:, 'Model'] = pd.Categorical(df['Model'], categories=model_names, ordered=True)

    colors = [Colors.TUD_BLUE, Colors.BU_GREEN1, Colors.ING_BLUE1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    plt.figure(figsize=(12, 6))

    labels = [metric_labels.get(metric, metric.replace('_', ' ').title()) for metric in metrics]

    for model_name in model_names:
        model_df = df[df['Model'] == model_name].reset_index()

        # Extract metric values and target values for the current model
        model_metric_values = model_df[metrics].apply(pd.to_numeric)

        # Apply data transformation
        if transformation == 'log':
            model_metric_values = model_metric_values.apply(np.log1p)
        elif transformation == 'sqrt':
            model_metric_values = model_metric_values.apply(np.sqrt)

        # Compute the correlation matrix
        correlation_matrix = model_metric_values.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, yticklabels=labels, annot=True, cmap=cmap, center=0, fmt='.3f', linewidths=0.5)

        # Get current x-tick positions and set the labels
        tick_positions = np.arange(len(correlation_matrix.columns)) + 0.4
        # Rotate y-tick labels
        plt.yticks(rotation=0, ha='right', va='center', fontsize=16)
        plt.xticks(ticks=tick_positions, labels=labels, rotation=45, ha='left', va='bottom', fontsize=16)
        plt.tick_params(axis='both', labelsize=16, labelbottom=False, bottom=False, top=False, labeltop=True)

    plt.tight_layout()

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath(f"evaluation/{filename}.png"), dpi=300, transparent=True)

    plt.show()


def plot_qq_residuals(residuals_normalized: np.ndarray, filename: str = "", show_plot: bool = False,
                      save_fig: bool = False):
    """
    Generates a QQ plot to visualize the distribution of residuals against a normal distribution.

    Args:
        residuals_normalized (np.ndarray): An array of normalized residuals from a regression model.
        filename (str, optional): The filename for saving the figure. Defaults to an empty string.
        show_plot (bool, optional): If True, displays the plot. Defaults to False.
        save_fig (bool, optional): If True, saves the figure as a PNG file. Defaults to False.

    Returns:
        None: The function generates and displays (or saves) a QQ plot.
    """
    fig, ax = plt.subplots()

    # Create QQ plot
    QQ = probplot(residuals_normalized, dist="norm", plot=ax)

    line = ax.get_lines()[1]
    line.set_linewidth(2)
    line.set_color(Colors.MED_RED)

    scatter = ax.get_lines()[0]
    scatter.set_linewidth(2)
    scatter.set_color(Colors.TUD_BLUE)
    scatter.set_markerfacecolor(Colors.TUD_BLUE)
    scatter.set_markersize(12.0)
    scatter.set_alpha(0.5)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    plt.xlabel("Theoretical Quantiles", fontsize=20, labelpad=10)
    plt.ylabel("Ordered Residuals", fontsize=20, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("")

    plt.plot([], [], color=Colors.MED_RED, lw=1.5, alpha=0.8, label='Cummulative Normal Probability')
    plt.scatter([], [], color=Colors.TUD_BLUE, lw=1.5, alpha=0.8, label='Residuals')

    plt.legend(
        fontsize=16,
        bbox_to_anchor=(0.5, -0.35),
        loc='upper center',
        borderaxespad=0.5,
        markerscale=1.5,
        labelspacing=0.5,
        handlelength=2.0,
        framealpha=0.4,
        ncol=2
    )

    # Display grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

    plt.tight_layout()

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath(f"model_adequacy/{filename}"), dpi=300, transparent=True,
                    bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_residuals_vs_fitted_values(externally_studentized_residuals: np.ndarray, metric: str,
                               y_pred: pd.Series, filename: str = "",
                               show_plot: bool = False, save_fig: bool = False):
    """
    Generates a residuals vs. fitted values plot to assess model fit and detect potential inadequacies.

    Args:
        externally_studentized_residuals (np.ndarray): The externally studentized residuals from a regression model.
        metric (str): The name of the metric being analyzed.
        y_pred (pd.Series): The predicted values from the regression model.
        filename (str, optional): The filename for saving the figure. Defaults to an empty string.
        show_plot (bool, optional): If True, displays the plot. Defaults to False.
        save_fig (bool, optional): If True, saves the figure as a PNG file. Defaults to False.

    Returns:
        None: The function generates and displays (or saves) the residuals vs. fitted values plot.
    """

    plt.figure(figsize=(8, 5))

    sns.residplot(
        x=y_pred,
        y=externally_studentized_residuals,
        lowess=True,
        scatter_kws={'alpha': 0.5, 'color': Colors.TUD_BLUE},
        line_kws={'color': Colors.MED_RED, 'lw': 1.5, 'alpha': 0.8})

    # Add a custom legend for the scatter and line
    plt.scatter([], [], color=Colors.TUD_BLUE, alpha=0.5, label="Residuals")
    plt.plot([], [], color=Colors.MED_RED, label="Lowess Line")

    plt.xlabel("X")
    plt.ylabel("Residuals")
    plt.legend(
        fontsize=16,
        bbox_to_anchor=(0.5, -0.25),
        loc='upper center',
        borderaxespad=0.5,
        markerscale=1.5,
        labelspacing=0.5,
        handlelength=2.0,
        framealpha=0.4,
        ncol=2
    )

    # Customize plot
    plt.ylabel("Residuals", fontsize=20, labelpad=10)
    plt.xlabel(f'Fitted values {metric}', fontsize=20, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath(f"model_adequacy/{filename}"), dpi=300, transparent=True,
                    bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_influence(models: list, metrics: dict, characteristics: dict, df_results: pd.DataFrame, eval_run: str, show_plot: bool = False,
                   save_fig: bool = False):
    """
    Generates bar plots to visualize the influence of various characteristics on model evaluation metrics.

    Args:
        models (list): A list of model names to generate influence plots for.
        metrics (dict): A dictionary mapping metric identifiers to their display names.
        df_results (pd.DataFrame): A DataFrame containing model results, including metrics, impacts, and p-values.
        show_plot (bool, optional): If True, displays the plot. Defaults to False.
        save_fig (bool, optional): If True, saves the plot as a PNG file. Defaults to False.

    Returns:
        None: The function generates and displays (or saves) bar plots for each model and metric.
    """

    # Define the p-value categorization function
    def categorize_p_value(p):
        if p <= 0.001:
            return Colors.TUD_BLUE  # Highly statistically significant
        elif p <= 0.01:
            return Colors.BU_GREEN3  # Statistically significant
        elif p <= 0.05:
            return Colors.BU_GREEN2  # Marginally significant
        else:
            return Colors.BU_GREEN1  # Not statistically significant

    for model in models:
        # Create the barplot
        fig, ax = plt.subplots(1, len(metrics), figsize=(20, 8))
        model_entry = df_results[df_results['model'] == model]

        for i, metric in enumerate(metrics.items()):
            if len(metrics) > 1:
                axis = ax[i]
            else:
                axis = ax

            # Extract the first row for the model you want to plot
            row = model_entry[model_entry['metric'] == metric[0]].iloc[0]

            # Create lists of metrics, impacts, and p_values
            impacts = [row[f"{characteristic}"] for characteristic in characteristics.keys()]
            p_values = [row[f"p_{characteristic}"] for characteristic in characteristics.keys()]

            # Assign colors based on p-value categorization
            colors = [categorize_p_value(p) for p in p_values]

            bars = axis.barh(characteristics.keys(), impacts, color=colors)

            # Customize plot
            axis.set_xlabel(f"Impact on {metric[1]}", fontsize=25, labelpad=10)
            axis.set_ylabel("")
            axis.set_title(f"adj. $R^2={row['adjusted_score']:.3f}$", fontsize=20)

            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('black')

            # Display grid
            axis.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

            if i == 0:
                # Customize y-tick labels
                custom_yticks = [f"{label_name}" for label_name in characteristics.values()]
                # Set the y-ticks for the specific subplot
                axis.set_yticks(range(len(characteristics)))
                axis.set_yticklabels(custom_yticks, fontsize=20)
            if i > 0:
                axis.set_yticklabels("", fontsize=15)

                # Customize ticks
            axis.tick_params(axis='x', labelsize=20)
            axis.axvline(0)

        # Add a legend for the p-value ranges
        legend_handles = [
            mpatches.Patch(color=Colors.TUD_BLUE, label="p ≤ 0.001"),
            mpatches.Patch(color=Colors.BU_GREEN3, label="p ≤ 0.01"),
            mpatches.Patch(color=Colors.BU_GREEN2, label="p ≤ 0.05"),
            mpatches.Patch(color=Colors.BU_GREEN1, label="Not significant"),

        ]
        fig.legend(
            handles=legend_handles,
            fontsize=16,
            bbox_to_anchor=(0.5, -0.1),
            loc="lower center",
            ncol=4,
            framealpha=0.4,
        )

        if save_fig:
            plt.savefig(PLOTS_DIRECTORY.joinpath(f"evaluation/evaluation_impact_{model}_{eval_run}.png"), dpi=300,
                        transparent=True, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()


def significance_test(df: pd.DataFrame, eval_run: str,
                      characteristics: dict, models: list, metrics: dict,
                      transformation: str = 'log', transform_target: str = False,
                      save_as_csv: bool = False, create_plots: bool = False, show_plots: bool = False,
                      save_figs: bool = False) -> pd.DataFrame:
    """
    Performs a significance test using OLS regression to analyze the influence of various characteristics on model performance metrics.

    Args:
        df (pd.DataFrame): The input DataFrame containing model evaluation data.
        characteristics (dict): A dictionary mapping characteristic names to their display labels.
        models (list): A list of model names to include in the analysis.
        metrics (dict): A dictionary mapping metric identifiers to their display names.
        transformation (str, optional): The transformation to apply to the data ('log', 'square', or None). Defaults to 'log'.
        transform_target (bool, optional): If True, applies transformation to the target metric values. Defaults to False.
        save_as_csv (bool, optional): If True, saves the results as a CSV file. Defaults to False.
        create_plots (bool, optional): If True, generates and displays various diagnostic plots. Defaults to False.
        show_plots (bool, optional): If True, displays the generated plots. Defaults to False.
        save_figs (bool, optional): If True, saves the generated plots as image files. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing model-wise regression results, including coefficients, p-values, R-squared, and adjusted R-squared values.
    """

    data = df[df['Model'].isin(models)].reset_index(drop=True).copy()
    sanitized_columns = {col: col.split('@')[0] for col in df.columns}
    data = data.rename(columns=sanitized_columns)

    metric_keys = list(metrics.keys())

    data[list(characteristics.keys())] = data[list(characteristics.keys())].apply(pd.to_numeric)
    data[metric_keys] = data[metric_keys].apply(pd.to_numeric)

    # Transformations
    if transformation == 'log':
        if transform_target:
            data[metric_keys] = data[metric_keys].apply(np.log10)
        log_characteristics = [characteristic for characteristic in characteristics.keys() if
                               not characteristic.startswith('degree')]
        data[log_characteristics] = data[log_characteristics].apply(np.log10)
    elif transformation == 'square':
        if transform_target:
            data[metric_keys] = data[metric_keys].apply(np.sqrt)
        log_characteristics = [characteristic for characteristic in characteristics.keys() if
                               not characteristic.startswith('degree')]
        data[log_characteristics] = data[log_characteristics].apply(np.sqrt)
    ## Centering
    data[list(characteristics.keys())] = data[list(characteristics.keys())].apply(lambda x: x - x.mean())

    # Iterate over the metrics to create the new columns
    df_new = pd.DataFrame()
    for model in models:
        for metric in metric_keys:
            df_new[f"{model}_{metric}"] = data[data['Model'] == model][metric].reset_index(drop=True)
        for characteristic in characteristics:
            df_new[characteristic] = data[data['Model'] == model][characteristic].reset_index(drop=True)
    train = df_new

    models_results = []
    for metric_key, metric_value in metrics.items():
        for idx, model in enumerate(models):
            X = train[list(characteristics.keys())]
            y = train[model + '_' + metric_key]
            # Define the formula for OLS regression
            formula_str_ml = y.name + ' ~ ' + '+'.join(list(characteristics.keys()))
            # Perform OLS regression using statsmodels
            model_ml = sm.ols(formula=formula_str_ml,
                              data=train[list(characteristics.keys()) + [model + '_' + metric_key]])
            fitted_ml = model_ml.fit(cov_type='HC1')  # Robust standard errors (HC1)

            print(fitted_ml.summary())
            print("Parameters: ", fitted_ml.params)
            print("R2: ", fitted_ml.rsquared)

            models_results.append({
                'model': model,
                'metric': metric_key,
                'score': fitted_ml.rsquared,
                'adjusted_score': fitted_ml.rsquared_adj,
                **fitted_ml.params.to_dict(),
                **fitted_ml.pvalues.rename(lambda x: 'p_' + x).to_dict()
            })

            if create_plots:
                # Plot Residuals vs. distribution
                influence = fitted_ml.get_influence()
                externally_studentized_residuals = influence.resid_studentized_external
                plot_qq_residuals(externally_studentized_residuals,
                                  filename=f"external_studentized_residuals_vs_probability_plot_{model}_{metric_key}_{eval_run}",
                                  show_plot=show_plots, save_fig=save_figs)

                # Plot Residuals vs. fitted values
                y_pred = fitted_ml.fittedvalues
                print(type(y_pred))
                plot_residuals_vs_fitted_values(externally_studentized_residuals, metric_value, y_pred,
                                           filename=f"external_studentized_residuals_vs_fitted_values_{model}_{metric_key}_{eval_run}.png",
                                           show_plot=show_plots, save_fig=save_figs)

    df_results = pd.DataFrame.from_dict(models_results)

    if create_plots:
        # Plot the influence
        plot_influence(models, metrics=metrics, characteristics=characteristics, df_results=df_results, eval_run=eval_run, show_plot=show_plots, save_fig=save_figs)

    if save_as_csv:
        df_results.to_csv(ASSET_DIRECTORY.joinpath(f"statistics_significance_tests/SignificanceTest-{eval_run}.csv"),
                          sep='\t', index=False)
    return df_results


"""
    Best / Worst users evaluation
"""

def plot_best_worst_characteristics(data: pd.DataFrame, model: str, characteristics: list, eval_run: str, num_datasets: int=176 ,save_fig: bool=False):
    # Filter the data for the selected model
    df = data[data['Model'] == model].copy()

    # Plot centroids as a boxplot
    fig, ax = plt.subplots(1, len(characteristics), layout="constrained", figsize=(16, 6))

    for i, characteristic in enumerate(characteristics):
        best_characteristics = []
        worst_characteristics = []

        axis = ax[i]
        for index, row in tqdm(df.iloc[:num_datasets].iterrows(), total=num_datasets, unit='rows'):
            if characteristic == 'node_degree':
                ylabel = 'AvgDeg'
                best_characteristics.append(row['best_users_node_degree_mean'])
                worst_characteristics.append(row['worst_users_node_degree_mean'])
            elif characteristic == 'degree_assort':
                ylabel = 'Assort'
                best_characteristics.append(row['degree_assort_best_users'])
                worst_characteristics.append(row['degree_assort_worst_users'])
            elif characteristic == 'average_coefficients':
                ylabel = 'AvgClustC'
                best_characteristics.append(row['average_clustering_coef_dot_best_users'])
                worst_characteristics.append(row['average_clustering_coef_dot_worst_users'])
            elif characteristic == 'average_popularity':
                ylabel = 'ARP'
                best_characteristics.append(row['best_users_mean_popularity_mean'])
                worst_characteristics.append(row['worst_users_mean_popularity_mean'])

        # Create a DataFrame for centroids
        df_centroids = pd.DataFrame({
            'Characteristic': np.concatenate([best_characteristics, worst_characteristics]),
            'Group': ['$Users^+$'] * len(best_characteristics) + ['$Users^-$'] * len(worst_characteristics)
        })

        custom_palette = [Colors.TUD_BLUE, Colors.BU_GREEN1]

        # Plot centroids as a boxplot
        sns.boxplot(data=df_centroids, x='Group', y='Characteristic', palette=custom_palette, hue='Group',
                    boxprops=dict(alpha=.3), ax=axis)
        sns.stripplot(data=df_centroids, x='Group', y='Characteristic', color=Colors.BU_GREEN3, alpha=0.6, jitter=True, ax=axis)

        # Customize plot
        axis.set_ylabel(ylabel, fontsize=25, labelpad=10)
        axis.set_xlabel("")

        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')

        # Display grid
        axis.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

        # Customize ticks
        axis.tick_params(axis='x', labelsize=20)
        axis.tick_params(axis='y', labelsize=20)

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath(f"evaluation/boxplot_best_worst_users_results_{model}_{eval_run}.png"), dpi=300, transparent=True)

    plt.ylabel(ylabel)
    plt.show()


"""
    QuickEval Prerequisites
"""

def get_prerequisits(eval_run):
    model_name_dict = {"AsymItemKNN": "ItemkNN", "AsymUserKNN": "UserkNN", "ALS": "ALS", "BPR": "BPR",
                       "LightGCN": "LightGCN", "UltraGCN": "UltraGCN", "SGL": "SGL", "XSimGCL": "XSimGCL", "Pop": "MostPop"}
    final_eval_df = get_evaluation_data(eval_run)
    model_list = final_eval_df['Model'].unique()
    filtered_model_dict = {key: value for key, value in model_name_dict.items() if key in model_list and key != "Pop"}

    return final_eval_df, filtered_model_dict, model_list