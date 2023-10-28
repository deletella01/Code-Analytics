import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def data_wrangler(filename, threshold=30000):
    """
    Read and preprocess a CSV file to create a cleaned DataFrame.

    Parameters:
    - filename (str): The name of the CSV file to be read.
    - threshold (int): The threshold for dropping columns with more than a set number of non-null values.

    Returns:
    - model_df (pandas.DataFrame): The cleaned DataFrame after preprocessing.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Drop columns with more than a set threshold of non-null values
    model_df = df.dropna(axis=1, thresh=threshold)

    # Fill null values in import and export quantities to depict negation
    float_columns = list(model_df.select_dtypes('float').columns)
    model_df[float_columns] = model_df[float_columns].fillna(0)

    # Drop rows with null values from `model_df`
    model_df.dropna(inplace=True)

    # Convert the 'Year' column to a string
    model_df["Year"] = model_df["Year"].astype(str)

    # Filter rows based on import and export quantity percentiles
    min_threshold_import, max_threshold_import = df['Importer reported quantity'].quantile([0.05, 0.99])
    min_threshold_export, max_threshold_export = df['Exporter reported quantity'].quantile([0.05, 0.99])

    model_df = df[
        (df['Importer reported quantity'] < max_threshold_import) &
        (df['Importer reported quantity'] > min_threshold_import) &
        (df['Exporter reported quantity'] < max_threshold_export) &
        (df['Exporter reported quantity'] > min_threshold_export)
    ]

    # Drop specific columns ('Origin' and 'Unit')
    drop_cols = ['Origin', 'Unit']
    model_df.drop(drop_cols, axis=1, inplace=True)

    # Drop any remaining rows with null values from `model_df`
    model_df.dropna(inplace=True)

    # Return the cleaned DataFrame
    return model_df
