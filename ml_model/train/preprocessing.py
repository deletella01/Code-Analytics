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

    # Convert the 'Year' column to a string
    model_df["Year"] = model_df["Year"].astype(str)
    
    # Filter rows where 'App' is not equal to 'N'
    model_df = model_df[model_df['App.'] != 'N']

    # Drop leaky features and unrequired features based on initial training run
    drop_cols = ['Year', 'Importer', 'Exporter', 'Importer reported quantity', 'Exporter reported quantity']
    model_df.drop(drop_cols, axis=1, inplace=True)

    # Drop any remaining rows with null values from `model_df`
    model_df.dropna(inplace=True)

    # Return the cleaned DataFrame
    return model_df
