import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from src.config import PLOTS_DIRECTORY, Colors

"""
    Evaluate Hyper-Parameter Search
"""

def plot_parameter_influence_knn(df: pd.DataFrame, parameters: list, metric: tuple,
                                  algorithms: list, custom_labels: list, type: str="line", save_fig: bool=False) -> None:
    """
    Plot the influence of parameters on a specified metric for selected algorithms in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing algorithms, parameters, and the metric.
        parameters (list): List of parameters to plot.
        metric (tuple): Tuple with the metric column name and label (e.g., ('test_ndcg@10', 'NDCG@10')).
        algorithms (list or None): List of algorithms to include in the plot (default: None, includes all algorithms).
        custom_labels (dict or None): Dictionary with algorithm names as keys and custom legend labels as values.

    Returns:
        None
    """

    # Define custom colors
    colors = {
        "asymitemknn": Colors.BU_GREEN1,
        "asymuserknn": Colors.TUD_BLUE,
    }

    # Filter dataframe for the selected algorithms
    if algorithms:
        df = df[df['algorithm'].isin(algorithms)]  # Select multiple algorithms

    # Calculate optimal number of rows and columns per plot
    number_of_params = len(parameters)
    max_cols = 4
    cols = min(number_of_params, max_cols)  # Limit columns to max_cols or fewer
    rows = (number_of_params + cols - 1) // cols  # Calculate the number of rows

    # Create the subplots
    # , layout="constrained"
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 6 * rows), layout="constrained")

    index = 0
    for row in range(rows):
        for col in range(cols):
            if index >= len(parameters):
                axis = ax[row, col] if rows > 1 else ax[col]
                axis.axis('off')
                continue

            # Extract the current parameter and metric for plotting
            param_name, param_label = parameters[index]
            metric_name, metric_label = metric

            axis = ax[row, col] if rows > 1 else ax[col]

            for algorithm in df['algorithm'].unique():
                algorithm_data = df[df['algorithm'] == algorithm]
                mean_value = algorithm_data.groupby(param_name)[metric_name].mean()

                if type == "line":
                    sns.lineplot(
                        x=mean_value.index,
                        y=mean_value.values,
                        ax=axis,
                        linewidth=2,
                        color=colors.get(algorithm, "black"),
                        legend=None  # Do not add legends for subplots
                    )
                elif type == "scatter":
                    # Plot the raw observations as scatter points
                    sns.scatterplot(
                        x=algorithm_data[param_name],  # Use raw observations of the parameter
                        y=algorithm_data[metric_name],  # Use the raw values for the metric
                        ax=axis,
                        color=colors.get(algorithm),  # Use the predefined color or default black
                        label=custom_labels.get(algorithm, algorithm) if custom_labels else algorithm,  # Use custom label if provided
                        alpha=0.7,  # Semi-transparent points
                        legend=None  # Do not add legends for subplots
                    )

            if param_name == "k":
                axis.set(xscale="log")

            axis.set_xlabel(f"{param_label}", fontsize=25)

            if index == 0 or index == 4:
                axis.set_ylabel(f"{metric[1]}", fontsize=25)
            else:
                axis.set_ylabel("")
                axis.tick_params(labelleft=False)
                axis.set_yticklabels([])

            axis.tick_params(axis='x', labelsize=20)
            axis.tick_params(axis='y', labelsize=20)

            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('black')

            axis.minorticks_on()
            axis.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

            index += 1

    # Create custom legend handles for consistent line styles and colors
    legend_handles = [
        Line2D([0], [0], color=colors[algorithm], lw=2, label=custom_labels.get(algorithm, algorithm) if custom_labels else algorithm)
        for algorithm in df['algorithm'].unique()
    ]

    # Add the legend with transparency
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),  # Adjusted position
        ncol=len(df['algorithm'].unique()),
        fontsize=20,
        framealpha=0.4  # Transparent legend background
    )

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath("hyper-parameter_tuning/hyper_setting_line_knn.png"), dpi=300, transparent=True)
    plt.show()

def plot_parameter_influence(df: pd.DataFrame, parameters: list, metric: str,
                             algorithm: list, type: str="line", save_fig: bool=False):
    """
    Plot the influence of parameters on a specified metric for selected algorithms in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing algorithms, parameters, and the metric.
        parameters (list): List of parameters to plot.
        metric (str): The column name of the metric to plot (e.g., 'test_ndcg@10').
        algorithm (list or None): List of algorithms to include in the plot (default: None, includes all algorithms).

    Returns:
        None
    """

    # Filter dataframe for the selected algorithm
    if algorithm:
        df = df[df['algorithm'] == algorithm]

    # Calculate optimal number of rows and columns per plot
    number_of_params = len(parameters)
    max_cols = 4
    cols = min(number_of_params, max_cols)  # Limit columns to max_cols or fewer
    rows = (number_of_params + cols - 1) // cols  # Calculate the number of rows

    # Create the subplots
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 6 * rows))

    index = 0
    # Loop through the parameters and plot
    for row in range(rows):
        for col in range(cols):
            if index >= len(parameters):
                axis = ax[row, col]
                axis.axis('off')
                continue
            # Extract the current parameter and metric for plotting
            param_name, param_label = parameters[index]
            metric_name, metric_label = metric

            if cols == 1:
                axis = ax
            else:
                if rows > 1:
                    axis = ax[row, col]
                else:
                    axis = ax[col]

            # Calculate the mean of the metric for the current parameter
            mean_value = df.groupby(param_name)[metric_name].mean()

            if type == "line":
                # Now plot the mean line using sns.lineplot to ensure consistency
                sns.lineplot(
                    x=mean_value.index,
                    y=mean_value.values,
                    ax=axis,
                    # label="Mean",
                    color=Colors.TUD_BLUE,
                    linewidth=2
                )
            elif type == "scatter":
                # Plot the data
                sns.scatterplot(
                    data=df,
                    x=parameters[index][0],  # Adjusting the index to select the correct parameter
                    y=metric[0],
                    ax=axis,
                    color=Colors.TUD_BLUE
                )

            if param_name in ["learning_rate", "regularization", "reg_weight", "ssl_weight", "lambda", "w1", "w3",
                              "negative_num", "negative_weight"]:
                axis.set(xscale="log")

            # Set the labels and titles using LaTeX-style formatting
            axis.set_xlabel(f"{param_label}", fontsize=25)

            # Only set y-label and y-ticks for the first plot
            if index == 0:
                axis.set_ylabel(f"{metric[1]}", fontsize=25)
            elif index == 4:
                axis.set_ylabel(f"{metric[1]}", fontsize=25)
            else:
                axis.set_ylabel("")
                axis.tick_params(labelleft=False)  # Disable y-ticks
                axis.set_yticklabels([])  # Explicitly clear y-tick labels

            # Increase the size of x-ticks and y-ticks
            axis.tick_params(axis='x', labelsize=20)
            axis.tick_params(axis='y', labelsize=20)
            # axis.legend(fontsize=20)

            # Show axis spines explicitly
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('black')

            # Add grid and axis lines
            axis.minorticks_on()
            axis.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')  # Grid lines
            axis.grid(True, which='minor', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')  # Grid lines

            # Add the mean value annotation on the plot (if needed)
            # for x, y in zip(mean_value.index, mean_value.values):
            #    axis.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)

            if index < len(parameters):
                index += 1

            if col > 0:
                axis.set_ylabel("")  # Clear the y-axis label for all but the first column

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Add more space at the bottom for x-labels

    if save_fig:
        plt.savefig(PLOTS_DIRECTORY.joinpath(f"hyper-parameter_tuning/hyper_setting_line_{algorithm}.png"), dpi=300, transparent=True)

    plt.show()
