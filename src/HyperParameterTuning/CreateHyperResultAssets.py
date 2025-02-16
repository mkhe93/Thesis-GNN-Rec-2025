import pandas as pd
import re
from src.config import HYPER_DIRECTORY
HYPER_RESULT_DIRECTORY = HYPER_DIRECTORY.joinpath("results")

# Function to parse a single .result file
def parse_result_file(file_path):
    results = []

    # Extract algorithm name from the filename
    file_name = str(file_path).split('/')[-1]
    algorithm_name = str(file_name).split('-')[0]

    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into blocks based on the parameter lines
    blocks = re.split(r'(?<=\n)\s*(?=\w+:)', content)

    for block in blocks:
        if not block.strip():
            continue

        # Extract the parameter line
        param_line_match = re.match(r'^([\w,:.\[\]eE\s+-]+)', block)
        if param_line_match:
            param_line = param_line_match.group(0)
            # Adjust regex to handle scalar values, scientific notation, and lists
            params = dict(re.findall(r'(\w+):(\[[^\]]+\]|[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?|[\d.e-]+)', param_line))
        else:
            continue

        # Extract metrics for valid results
        valid_result_match = re.search(r'Valid result:\s*((?:\w+@10\s*:\s*[0-9\.]+\s*)+)', block)
        valid_metrics = {}
        if valid_result_match:
            valid_metrics = dict(re.findall(r'(\w+@10)\s*:\s*([0-9\.]+)', valid_result_match.group(1)))

        # Extract metrics for test results
        test_result_match = re.search(r'Test result:\s*((?:\w+@10\s*:\s*[0-9\.]+\s*)+)', block)
        test_metrics = {}
        if test_result_match:
            test_metrics = dict(re.findall(r'(\w+@10)\s*:\s*([0-9\.]+)', test_result_match.group(1)))

        # Combine parameters, valid metrics, test metrics, algorithm name, and dataset name into a single row
        result = {
            'algorithm': algorithm_name,
            **params,
            **{f'valid_{k}': v for k, v in valid_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()}
        }
        results.append(result)

    return results

# Function to load multiple .result files and combine them into a DataFrame
def load_result_files(file_paths):
    all_results = []
    for file_path in file_paths:
        all_results.extend(parse_result_file(file_path))
    return pd.DataFrame(all_results)

# Function to print top 3 settings based on ndcg@10 for each algorithm and dataset
def print_top_ndcg(df):
    print("Top settings based on ndcg@10 for each algorithm and dataset:")
    grouped = df.groupby(['algorithm', 'dataset'])

    for (algorithm, dataset), group in grouped:
        # Filter columns that do not start with 'valid_' or 'test_' and are not in the exclusion list
        filtered_columns = [col for col in group.columns if
                            not (col.startswith('valid_') or col.startswith('test_'))]
        group = group[filtered_columns + ['test_ndcg@10']]  # Keep relevant columns

        print(f"\nAlgorithm: {algorithm}")
        top_ndcg = group.sort_values(by='test_ndcg@10', ascending=False).head(3)
        print(top_ndcg)

    print("\nTop settings based on ndcg@10 for the entire dataset:")
    filtered_columns = [col for col in df.columns if not (col.startswith('valid_') or col.startswith('test_'))]
    df_filtered = df[filtered_columns + ['test_ndcg@10']]  # Keep relevant columns
    top_ndcg_all = df_filtered.sort_values(by='test_ndcg@10', ascending=False).head(10)
    print(top_ndcg_all)

def export_top_ndcg_to_latex(df, top_k=3, output_file="top_ndcg.tex"):
    """
    Exports the top 3 settings based on `ndcg@10` for each algorithm as LaTeX tables.

    :param df: DataFrame containing the data.
    :param output_file: Name of the output LaTeX file.
    """
    print("Exporting top 3 settings based on ndcg@10 for each algorithm as LaTeX tables.")

    grouped = df.groupby(['algorithm'])  # Group by algorithm
    latex_tables = []

    # Define the columns to include based on the filter
    params = [col for col in df.columns if
              not (col.startswith('valid_') or col.startswith('test_')) and col not in [
                  'algorithm', 'dataset', 'valid_ndcg@10', 'test_ndcg@10']]

    for algorithm, group in grouped:
        #print(f"Processing top 3 settings for algorithm: {algorithm}")

        # Filter out columns with all NaN values for this algorithm
        relevant_columns = [col for col in params if not group[col].isna().all()]
        columns_to_keep = ['algorithm', 'test_ndcg@10'] + relevant_columns
        filtered_group = group[columns_to_keep]

        # Sort by `test_ndcg@10` and get the top 3 rows
        top_ndcg = filtered_group.sort_values(by='test_ndcg@10', ascending=False).head(top_k)

        # Convert to LaTeX table
        latex_table = top_ndcg.to_latex(
            index=False,
            caption=f"Top {top_k} settings for algorithm: {algorithm}",
            label=f"tab:{algorithm}_top{top_k}",
            float_format="%.7f"  # Ensures consistent numeric formatting
        )
        latex_tables.append(latex_table)

    # Write all tables to a LaTeX file
    with open(output_file, "w") as f:
        for table in latex_tables:
            f.write(table + "\n\n")

    print(f"LaTeX tables have been written to {output_file}.")


def export_param_ranges_by_algorithm_to_latex(df, output_file="param_ranges_by_algorithm.tex"):
    """
    Exports the range of parameter values for each algorithm as a LaTeX table.

    :param df: DataFrame containing algorithm-specific parameter values.
    :param output_file: Name of the output LaTeX file.
    """
    print("Exporting parameter value ranges for each algorithm as a LaTeX table.")

    # Define the parameter columns (filter out metadata and metric columns)
    param_columns = [col for col in df.columns if
                     not (col.startswith('valid_') or col.startswith('test_')) and col not in [
                         'algorithm', 'dataset', 'valid_ndcg@10', 'test_ndcg@10']]

    latex_tables = []

    # Group by algorithm and calculate parameter ranges for each algorithm
    grouped = df.groupby('algorithm')

    for algorithm, group in grouped:
        #print(f"\nProcessing parameters for algorithm: {algorithm}")

        # Filter out columns with all NaN values for this algorithm
        group = group[param_columns]  # Start with all parameter columns
        relevant_params = [col for col in group.columns if not group[col].isna().all()]  # Remove all-NaN columns
        group = group[relevant_params]  # Keep only relevant columns

        # Calculate ranges for each parameter
        param_ranges = {}
        for param in relevant_params:
            unique_values = sorted(group[param].dropna().unique())  # Drop NaNs and sort unique values
            if len(unique_values) > 1:
                # Display range as a set for multiple values
                param_ranges[param] = "{" + ", ".join(map(str, unique_values)) + "}"
            else:
                # Single fixed value
                param_ranges[param] = str(unique_values[0])

        # Create a DataFrame to store parameter ranges for the current algorithm
        param_ranges_df = pd.DataFrame(
            list(param_ranges.items()),
            columns=["Parameter", "Values"]
        )

        # Convert to LaTeX table
        latex_table = param_ranges_df.to_latex(
            index=False,
            caption=f"Parameter value ranges for algorithm: {algorithm}",
            label=f"tab:param_ranges_{algorithm}"
        )
        latex_tables.append(latex_table)

    # Write all tables to a LaTeX file
    with open(output_file, "w") as f:
        for table in latex_tables:
            f.write(table + "\n\n")

    print(f"Parameter ranges for all algorithms have been exported to {output_file}.")

# Example usage
if __name__ == "__main__":

    hyper_search_results = ["asymitemknn-hyper-RO.result",
                            "asymuserknn-hyper-RO.result",
                            "als-hyper-RO.result",
                            "bpr-hyper-RO.result",
                            "lightgcn-hyper-RO.result",
                            "ultragcn-hyper-RO.result",
                            "sgl-hyper-RO.result",
                            "xsimgcl-hyper-RO.result"]

    file_paths = [HYPER_RESULT_DIRECTORY.joinpath(result) for result in hyper_search_results]

    df = load_result_files(file_paths)

    # Convert numeric columns to appropriate types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    top = 5
    export_top_ndcg_to_latex(df, top_k=top, output_file=f"results/Top{top}NDCGParams.tex")
    df.to_csv(HYPER_RESULT_DIRECTORY.joinpath("HyperSearchResults.csv"), index=False, sep="\t")
