from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging


def log_model_performance(output_directory, model_type, pos_class_prob_values, accuracy, specificity, sensitivity,
                          feature_importance_df, tp, tn, fp, fn, clf, n_positive, n_negative):

    """
    Logs the performance metrics of a machine learning model and saves the results to files,
    including a probability distribution plot and a text log file.
    This function is designed to provide detailed insights into the model's performance and feature importance.

    :param str output_directory:
        The directory where output files, including performance logs and plots, will be saved.
        A subdirectory named ML/ is created if it does not already exist.
    :param str model_type:
        Descriptive name of model (divided_vs_non_divided, non_divided, divided)
    :param np.ndarray pos_class_prob_values:
        Predicted probabilities for the positive class from the trained model.
    :param float accuracy:
        Overall accuracy of the model, calculated as (TP + TN) / (TP + TN + FP + FN).
    :param float specificity:
        Model specificity, calculated as TN / (TN + FP).
    :param float sensitivity:
        Model sensitivity (recall), calculated as TP / (TP + FN).
    :param pandas.DataFrame feature_importance_df:
        DataFrame containing feature names and their respective weights or importances.
        Empty if the model does not support feature importance extraction.
    :param int tp:
        Count of true positive predictions.
    :param int tn:
        Count of true negative predictions.
    :param int fp:
        Count of false positive predictions.
    :param int fn:
        Count of false negative predictions.
    :param str clf:
        Name of the classifier used for the analysis.
    :param int n_positive:
        Total number of positive samples (links exist in dataframe) in the dataset.
    :param int n_negative:
        Total number of negative samples (synthetic links between targets and neighbors of sources) in the dataset.

    **Returns**:
        None
    """

    output_directory = output_directory + '/ML/'
    os.makedirs(output_directory, exist_ok=True)

    # Suppress Matplotlib font debug warnings
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Set a default font to avoid font search
    plt.rcParams['font.family'] = 'Arial'

    # Create a density plot with seaborn
    plt.figure(figsize=(8, 6))
    sns.kdeplot(pos_class_prob_values, fill=True, color='skyblue', alpha=0.6)

    # Limit x-axis from 0 to 1
    plt.xlim(0, 1)

    # Add labels and title
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(model_type + '.' + clf + 'probability distribution for correct links with high chance')
    plt.savefig(output_directory + '/' + model_type + '.' + clf + 'probability_distribution.jpg', dpi=600)
    plt.close()

    with open(output_directory + '/' + model_type + '.' + clf + '.performance.log.txt', "w") as file:
        # Write the outputs to the file
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Specificity: {specificity}\n")
        file.write(f"Sensitivity: {sensitivity}\n")
        file.write(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}\n")
        file.write(f"Total number of samples in class 1: {n_positive}\n")
        file.write(f"Total number of samples in class 2: {n_negative}\n")
        if feature_importance_df.shape[0] > 0:
            file.write(f"Feature Weights: {feature_importance_df}\n")


def train_model(merged_df, feature_list, columns_to_scale, model_type, output_directory, clf, n_cpu):

    """

    Trains a machine learning model using the specified classifier and evaluates its performance.
    The function applies preprocessing, trains the model, and calculates various performance metrics such as
    accuracy, specificity, and sensitivity.

    :param pandas.DataFrame merged_df:
        The input DataFrame containing the features and target labels for training the model.
        The target labels are expected to be in a column named `label`.
    :param list feature_list:
        A list of column names in merged_df representing the features used for training.
    :param list columns_to_scale:
        A list of feature columns in merged_df that should be scaled using StandardScaler.
    :param str model_type:
        Descriptive name of model (divided_vs_non_divided, non_divided, divided)
    :param str output_directory:
        The directory where output files, including performance logs and plots, will be saved.
    :param str clf:
        The name of the classifier to use. Supported classifiers include:
            - 'LogisticRegression'
            - 'GaussianProcessClassifier'
            - 'C-Support Vector Classifier'
    :param int n_cpu:
        Number of CPU cores to use for parallel processing in the classifier.

    Returns:
        sklearn.pipeline.Pipeline

        :returns: A trained machine learning pipeline containing the preprocessing steps and the classifier.
    """

    clf_dict = {
                'LogisticRegression': LogisticRegression(random_state=42, n_jobs=n_cpu, class_weight='balanced'),
                'GaussianProcessClassifier': GaussianProcessClassifier(random_state=42, n_jobs=n_cpu),
                'C-Support Vector Classifier': SVC(random_state=42, probability=True, class_weight='balanced')}

    # Define features and target
    x_dat = merged_df[feature_list]
    y_dat = merged_df['label']

    n_positive = merged_df.loc[merged_df['label'] == 'positive'].shape[0]
    n_negative = merged_df.loc[merged_df['label'] == 'negative'].shape[0]

    # Create a ColumnTransformer to apply StandardScaler to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'
    )

    # Define the pipeline with SMOTE
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_dict[clf])
    ])

    x_train = x_dat.copy()
    y_train = y_dat.copy()

    train_dat = x_train.copy()
    train_dat['real_label'] = y_train.copy()

    # Train the Logistic Regression model on the scaled data
    # Fit the pipeline on the training data
    pipeline.fit(x_train, y_train)

    # Predict on the test set
    y_predicted_train = pipeline.predict(x_train)

    y_prob_train = pipeline.predict_proba(x_train)[:, 1]

    # Convert the numpy array to a DataFrame
    y_predicted_train_df = pd.DataFrame(y_prob_train, index=x_train.index, columns=['probability'])

    train_dat = pd.concat([train_dat, y_predicted_train_df], axis=1)
    train_dat_class_positive = train_dat.loc[train_dat['real_label'] == 'positive']
    pos_class_prob_values = train_dat_class_positive['probability'].values

    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_predicted_train)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_train, y_predicted_train)

    # Calculate specificity and sensitivity from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Get the weights of features
    model = pipeline.named_steps['classifier']

    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names from the ColumnTransformer
    # Depending on your transformers, the method to get feature names might differ
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Example for older versions or different transformers
        feature_names = preprocessor.transformers_[0][1].get_feature_names_out()

    # Extract feature weights from the logistic regression model
    try:
        feature_weights = model.coef_[0]
    except AttributeError:
        try:
            # Try to get feature importance for models
            feature_weights = model.feature_importances_[0]
        except AttributeError:
            # For models like GaussianProcessClassifier, set feature weights to NaN
            feature_weights = np.nan

    if str(feature_weights) != 'nan':
        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': feature_weights
        })
    else:
        feature_importance_df = pd.DataFrame()

    # Report results
    log_model_performance(output_directory, model_type, pos_class_prob_values, accuracy, specificity, sensitivity,
                          feature_importance_df, tp, tn, fp, fn, clf, n_positive, n_negative)

    return pipeline
