import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import re
from src.config import Colors, EPOCH_TUNING_DIRECTORY, PLOTS_DIRECTORY

"""
    Evaluate Epoch Tuning
"""

def plot_training_results(
        log_files,
        plot_mean=False,
        plot_ndgc_alone=False,
        line_width=2,
        save_fig=False
):
    """
    Parses log files and plots training results (loss and NDCG@10).

    Parameters:
        log_files (dict): A dictionary with model names as keys and log file paths as values.
        plot_mean (bool): Whether to plot the mean across runs or individual runs.
        colors (dict): Optional list of colors for the plot.
        line_width (int or float): Line width for the plot lines.
    """

    # Function to parse a single log file
    def parse_log_file(file_path, model_name):

        file_path = EPOCH_TUNING_DIRECTORY.joinpath("EpochEvaluation").joinpath(file_path)

        # Regular expressions for extracting information
        training_pattern = re.compile(r"epoch (\d+) training \[time: ([\d.]+)s, train loss: ([\d.]+)\]")
        training_pattern_xsim = re.compile(r"epoch (\d+) training \[time: ([\d.]+)s, train_loss1: ([\d.]+)")
        ndcg_pattern = re.compile(r"ndcg@10 : ([\d.]+)")

        all_data = []
        run_id = 0
        with open(file_path, "r") as f:
            for line in f:
                # Match training info
                training_match = training_pattern.search(line)
                training_xsimgcl_match = training_pattern_xsim.search(line)
                if training_match or training_xsimgcl_match:
                    if model_name == "XSimGCL":
                        epoch, time, train_loss = training_xsimgcl_match.groups()
                    else:
                        epoch, time, train_loss = training_match.groups()
                    epoch = int(epoch) + 1

                    # Calculate run_id and adjusted epoch#
                    if epoch == 1:
                        run_id += 1

                    all_data.append({
                        "model": model_name,
                        "run_id": run_id,
                        "epoch": epoch,
                        "time": float(time),
                        "train_loss": float(train_loss),
                        "ndcg@10": None,
                    })

                # Match NDCG@10 and update the last entry
                ndcg_match = ndcg_pattern.search(line)
                if ndcg_match and all_data:
                    all_data[-1]["ndcg@10"] = float(ndcg_match.group(1))

        return pd.DataFrame(all_data)

    colors = {
        "ALS": Colors.TUD_BLUE,
        "BPR": Colors.MN_GREEN,
        "LightGCN": Colors.LEH_ORANGE,
        "UltraGCN": Colors.ING_BLUE,
        "SGL": Colors.BU_GREEN3,
        "XSimGCL": Colors.MED_RED
    }

    # Parse log files and combine data
    df_combined = pd.concat(
        [parse_log_file(file_path, model_name) for model_name, file_path in log_files.items()],
        ignore_index=True,
    )

    # Create custom legend handles for consistent line styles and colors
    legend_handles = [
        Line2D([0], [0], color=colors[model], lw=2, label=model)
        for model in df_combined["model"].unique()
    ]

    # Plotting
    plt.figure(figsize=(14, 7))
    num_models = len(df_combined["model"].unique())

    if num_models == 1 and not plot_ndgc_alone:
        if plot_mean:
            # Plot Mean Train Loss
            plt.subplot(1, 2, 1)
            for model, df_model in df_combined.groupby("model"):
                df_loss = df_model.groupby("epoch")["train_loss"].mean()
                sns.lineplot(data=df_loss, label=f"{model} Mean Train Loss", linewidth=line_width,
                             color=colors.get(model, "black"))
            plt.ylabel("Train Loss", fontsize=25, labelpad=10)
            plt.xlabel("Epoch", fontsize=25, labelpad=10)

            ax = plt.gca()  # Get current axes
            for spine in ax.spines.values():
                spine.set_edgecolor('black')  # Set the color of the border
                spine.set_linewidth(1)  # Set the thickness of the border

            # Customize ticks
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            # plt.minorticks_on()
            plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

            # Add the legend with transparency
            plt.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),  # Adjusted position
                ncol=num_models,
                fontsize=20,
                framealpha=0.4  # Transparent legend background
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            # Plot Mean NDCG@10
            plt.subplot(1, 2, 2)
            for model, df_model in df_combined.groupby("model"):
                df_ndcg = df_model.dropna(subset=["ndcg@10"]).groupby("epoch")["ndcg@10"].mean()
                sns.lineplot(data=df_ndcg, label=f"{model} Mean NDCG@10", linewidth=line_width,
                             color=colors.get(model, "black"))
            plt.ylabel("NDCG@10", fontsize=25, labelpad=10)
            plt.xlabel("Epoch", fontsize=25, labelpad=10)

            ax = plt.gca()  # Get current axes
            for spine in ax.spines.values():
                spine.set_edgecolor('black')  # Set the color of the border
                spine.set_linewidth(1)  # Set the thickness of the border

            ymax = round(df_combined["ndcg@10"].max(),3)
            plt.ylim(bottom=0, top=ymax*1.1)

            # Customize ticks
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            # plt.minorticks_on()
            plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

            # Add the legend with transparency
            plt.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),  # Adjusted position
                ncol=num_models,
                fontsize=20,
                framealpha=0.4  # Transparent legend background
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)


        else:
            # Plot Train Loss for individual runs
            plt.subplot(1, 2, 1)
            for model, df_model in df_combined.groupby("model"):
                for run_id, df_run in df_model.groupby("run_id"):
                    sns.lineplot(data=df_run, x="epoch", y="train_loss", label=f"{model} Run {run_id} Train Loss",
                                 alpha=0.5, linewidth=line_width, color=colors.get(model, "black"))
            plt.ylabel("Train Loss", fontsize=25, labelpad=10)
            plt.xlabel("Epoch", fontsize=25, labelpad=10)

            ax = plt.gca()  # Get current axes
            for spine in ax.spines.values():
                spine.set_edgecolor('black')  # Set the color of the border
                spine.set_linewidth(1)  # Set the thickness of the border

            # Customize ticks
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            # plt.minorticks_on()
            plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

            # Add the legend with transparency
            plt.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),  # Adjusted position
                ncol=num_models,
                fontsize=20,
                framealpha=0.4  # Transparent legend background
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            # Plot NDCG@10 for individual runs
            plt.subplot(1, 2, 2)
            for model, df_model in df_combined.groupby("model"):
                for run_id, df_run in df_model.groupby("run_id"):
                    df_run = df_run.dropna(subset=["ndcg@10"])
                    sns.lineplot(data=df_run, x="epoch", y="ndcg@10", label=f"{model} Run {run_id} NDCG@10", alpha=0.5,
                                 linewidth=line_width, color=colors.get(model, "black"))
            plt.ylabel("NDCG@10", fontsize=25, labelpad=10)
            plt.xlabel("Epoch", fontsize=25, labelpad=10)

            ax = plt.gca()  # Get current axes
            for spine in ax.spines.values():
                spine.set_edgecolor('black')  # Set the color of the border
                spine.set_linewidth(1)  # Set the thickness of the border

            ymax = round(df_combined['ndcg@10'].max(),3)
            plt.ylim(bottom=0, top=ymax*1.1)

            # Customize ticks
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            # plt.minorticks_on()
            plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

            # Add the legend with transparency
            plt.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),  # Adjusted position
                ncol=num_models,
                fontsize=20,
                framealpha=0.4  # Transparent legend background
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

    elif num_models >= 2 or plot_ndgc_alone:
        if plot_mean:
            # Plot Mean NDCG@10 for all models
            for model, df_model in df_combined.groupby("model"):
                df_ndcg = df_model.dropna(subset=["ndcg@10"]).groupby("epoch")["ndcg@10"].mean()
                sns.lineplot(data=df_ndcg, label=f"{model} Mean NDCG@10", linewidth=line_width,
                             color=colors.get(model, "black"))
            plt.ylabel("NDCG@10", fontsize=25, labelpad=10)
        else:
            # Plot NDCG@10 for individual runs
            for model, df_model in df_combined.groupby("model"):
                for run_id, df_run in df_model.groupby("run_id"):
                    df_run = df_run.dropna(subset=["ndcg@10"])
                    sns.lineplot(data=df_run, x="epoch", y="ndcg@10", label=f"{model} Run {run_id} NDCG@10",
                                 color=colors.get(model, "black"), alpha=0.5, linewidth=line_width)
            plt.ylabel("NDCG@10", fontsize=25, labelpad=10)

        plt.xlabel("Epoch", fontsize=25, labelpad=10)

        ax = plt.gca()  # Get current axes
        for spine in ax.spines.values():
            spine.set_edgecolor('black')  # Set the color of the border
            spine.set_linewidth(1)  # Set the thickness of the border

        # Customize ticks
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # plt.minorticks_on()
        plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

        # Add the legend with transparency
        plt.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),  # Adjusted position
            ncol=num_models,
            fontsize=20,
            framealpha=0.4  # Transparent legend background
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)

    if save_fig:
        if num_models == 1:
            if plot_mean:
                plt.savefig(PLOTS_DIRECTORY.joinpath(f"epochs_tuning/epoch_mean_evaluation_{df_combined["model"].unique()[0]}.png"), dpi=300,
                            transparent=True, bbox_inches='tight')
            else:
                plt.savefig(PLOTS_DIRECTORY.joinpath(f"epochs_tuning/epoch_single_evaluation_{df_combined["model"].unique()[0]}.png"), dpi=300,
                            transparent=True, bbox_inches='tight')
        else:
            plt.savefig(
                PLOTS_DIRECTORY.joinpath(f"epochs_tuning/epoch_mean_evaluation_{'_'.join(model for model in df_combined["model"].unique())}.png"),
                dpi=300, transparent=True, bbox_inches='tight')

    plt.show()