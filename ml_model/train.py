import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier 
from category_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from processing import data_wrangler
warnings.filterwarnings('ignore')


def split_data():
    """
    Load data, split it into features (X) and target (y), and create training and testing sets.

    Returns:
    - X_train (pandas.DataFrame): The features for the training set.
    - X_test (pandas.DataFrame): The features for the testing set.
    - y_train (pandas.Series): The target values for the training set.
    - y_test (pandas.Series): The target values for the testing set.
    """

    # Load and preprocess the data using the `data_wrangler` function
    model_df = data_wrangler("comptab_2018-01-29 16_00_comma_separated.csv")
    
    # Split the data into features (X) and the target (y)
    X = model_df.drop("App.", axis=1)
    y = model_df["App."]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Return the training and testing sets
    return X_train, X_test, y_train, y_test

def resample():
    """
    Perform random oversampling on the training data to address class imbalance.

    Returns:
    - X_train_over (pandas.DataFrame): The oversampled feature data.
    - y_train_over (pandas.Series): The oversampled target data.
    """

    # Obtain the training data using the `split_data` function
    X_train, y_train = split_data()[0], split_data()[2]

    # Perform random oversampling on the training data
    X_train_over, y_train_over = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

    # Return the oversampled training data
    return X_train_over, y_train_over

def build_model(voting="hard"):
    """
    Build a Voting Classifier model with specified voting method and a set of base estimators.

    Parameters:
    - voting (str, optional): The voting method for the Voting Classifier. Default is "hard".

    Returns:
    - model (sklearn.pipeline.Pipeline): A pipeline containing an OrdinalEncoder and the Voting Classifier.
    """

    # Define a list of base estimators
    estimator = []
    estimator.append(('LR', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)))
    estimator.append(('SVC', SVC(gamma='auto', probability=True)))
    estimator.append(('DTC', DecisionTreeClassifier(max_depth=10)))
    estimator.append(('RFC', RandomForestClassifier(random_state=42)))
    estimator.append(('GBC', GradientBoostingClassifier()))

    # Create a Voting Classifier with the specified voting method
    model = make_pipeline(
        OrdinalEncoder(),
        VotingClassifier(estimators=estimator, voting=voting)
    )
    
    # Return the built model
    return model

def main():
    """
    Train a machine learning model, generate a classification report, save the report and model, and plot feature importance.

    Returns:
    - None
    """

    print("Model pipeline started...")

    # Build the machine learning model
    model = build_model()
    
    # Split the data into training and testing sets using the `split_data` function
    X_train, X_test, y_train, y_test = split_data()

    # Resample the training data to address class imbalance
    X_train_over, y_train_over = resample()

    # Fit the model with the oversampled data
    model.fit(X_train_over, y_train_over)
    
    # Specify the file path where you want to save the model
    model_path = f'model_{pd.Timestamp.now().isoformat()}.pkl'

    # Save the model to the file using pickle
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # joblib.dump(model, model_path)
    
    print(f'Model saved to {model_path}')
    
    # Generate the classification report
    report = classification_report(y_test, model.predict(X_test))

    # Define the file path where you want to save the report
    txt_path = "classification_report.txt"

    # Write the report to a file
    with open(txt_path, 'w') as file:
        file.write(report)
    
    print(f"Classification report has been saved to {txt_path}")
    
    # Plot feature importance for each base model in the ensemble
    for model in model.named_steps["votingclassifier"].estimators_:
        if hasattr(model, 'feature_importances_'):
            feature_importances_model = model.feature_importances_
        
    # Get feature names from training data
    features = model.feature_names_in_

    # Create a series with feature names and importances
    feat_imp = pd.Series(feature_importances_model, index=features).sort_values(ascending=False)
    
    # Create a horizontal bar plot for feature importance
    plt.figure(figsize=(8,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, orient='h',color=sns.color_palette()[0])
    plt.xlabel("Gini Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()

    # Specify the file path and format for saving the plot
    png_path = "feature_importance.png"  # You can choose the file format (e.g., .png, .jpg, .svg)

    # Save the plot to the specified file
    plt.savefig(png_path)

    # Print a confirmation message
    print(f"Feature importance has been saved to {png_path}")
    
if __name__ == "__main__":

    main()



    





