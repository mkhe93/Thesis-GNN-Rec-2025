import json
import pandas as pd
import ast

import os
from pathlib import Path

# Define paths
PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_DIRECTORY.joinpath("data")

SRC_DIRECTORY = PROJECT_DIRECTORY.joinpath("src")
ASSET_DIRECTORY = SRC_DIRECTORY.joinpath("assets")
PLOTS_DIRECTORY = ASSET_DIRECTORY.joinpath("plots")

HYPER_DIRECTORY = SRC_DIRECTORY.joinpath("HyperParameterTuning")
EPOCH_TUNING_DIRECTORY = SRC_DIRECTORY.joinpath("EpochTuning")
TESTRUN_DIRECTORY = SRC_DIRECTORY.joinpath("TestRuns")
EVALUATION_DIRECTORY = SRC_DIRECTORY.joinpath("Evaluation")
CONFIG_DIRECTORY = SRC_DIRECTORY.joinpath("config/config_files")

# Colors for plots
class Colors:
    """Defines custom color constants in RGB format (normalized to [0,1])."""
    TUD_BLUE = (0 / 255, 48 / 255, 93 / 255)
    BU_GREEN1 = (138 / 255, 203 / 255, 193 / 255)
    BU_GREEN2 = (0 / 255, 172 / 255, 169 / 255)
    BU_GREEN3 = (0 / 255, 131 / 255, 141 / 255)
    ING_BLUE1 = (132 / 255, 207 / 255, 237 / 255)
    ING_BLUE2 = (0 / 255, 161 / 255, 217 / 255)
    ING_BLUE3 = (0 / 255, 119 / 255, 174 / 255)
    ING_BLUE4 = (0 / 255, 105 / 255, 180 / 255)

    MN_GREEN = (148 / 255, 195 / 255, 86 / 255)
    BU_GREEN = (138 / 255, 203 / 255, 193 / 255)
    ING_BLUE = (132 / 255, 207 / 255, 237 / 255)
    MED_RED = (205 / 255, 65 / 255, 44 / 255)
    LEH_ORANGE = (247 / 255, 169 / 255, 65 / 255)


# Function to process columns
def json_dumps_user_columns(df: pd.DataFrame):
    for col in df.columns:
        if col.startswith('best_user_') or col.startswith('worst_user_'):
            # Apply transformation for each row in the selected columns
            df [col] = df[col].apply(json.dumps)
    return df

# Function to process columns
def json_loads_user_columns(df: pd.DataFrame):
    for col in df.columns:
        if col.startswith('best_user_') or col.startswith('worst_user_'):
            # Apply transformation for each row in the selected columns
            df [col] = df[col].apply(json.loads)
    return df

# Function to process columns
def process_user_columns(df):
    for col in df.columns:
        if col.startswith('best_user_') or col.startswith('worst_user_') or col.startswith('clustering_coefficients'):
            # Apply transformation for each row in the selected columns
            df[col] = df[col].apply(ast.literal_eval)
    return df