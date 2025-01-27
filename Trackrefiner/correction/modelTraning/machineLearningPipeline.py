from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
import copy


def draw_kde_plot(pos_class_prob_values_dict, model_type, clf, output_directory, dsc):

    """
    Draws and saves a KDE plot for the probability distributions of positive class probabilities across different
    trainings (Training on all data, Training on train split, Evaluation on test split).

    :param dict pos_class_prob_values_dict:
        A dictionary where keys represent training type and values are arrays of
        positive class probabilities for the respective class.
    :param str model_type:
        Descriptive name of the model type (e.g., division_vs_continuity).
    :param str clf:
        Name of the classifier used.
    :param str output_directory:
        Directory where the plot will be saved.
    :param str dsc:
        Description for the data being plotted.

    Returns:
        None
    """

    # Suppress Matplotlib font debug warnings
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Create a density plot with seaborn
    plt.figure(figsize=(8, 6))

    # Use a color palette for distinct colors
    colors = sns.color_palette('tab10', len(pos_class_prob_values_dict))

    for i, (key, values) in enumerate(pos_class_prob_values_dict.items()):
        sns.kdeplot(values, fill=True, color=colors[i], alpha=0.5, label=key)

    # Limit x-axis from 0 to 1
    plt.xlim(0, 1)

    # Add labels and title
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(f"Probability Distribution for {dsc}")
    # Add legend with a title
    plt.legend(title="Classes")

    output_file = f"{output_directory}/{model_type}_{clf}_{dsc.lower().replace(' ', '_')}_distribution.jpg"
    plt.savefig(output_file, dpi=600)
    plt.close()


def log_model_performance(output_directory, model_type, pos_class_prob_values_dict, pref_measures_dict,
                          feature_importance_df, clf, num_full_pos_neg_class_samples, dsc, clas_names):
    """
    Logs the performance metrics of a machine learning model and saves the results to files,
    including a probability distribution plot and a text log file.

    :param str output_directory:
        Directory where output files, including performance logs and plots, will be saved.
    :param str model_type:
        Descriptive name of model (division_vs_continuity, continuity, division)
    :param dict pos_class_prob_values_dict:
        Dictionary containing positive class probabilities for different sets (Training on all data,
        Training on train split, Evaluation on test split).
    :param dict pref_measures_dict:
        Dictionary of performance measures for each dataset.
    :param pandas.DataFrame feature_importance_df:
        DataFrame containing feature names and their importances or weights.
    :param str clf:
        Name of the classifier used.
    :param dict num_full_pos_neg_class_samples:
        Dictionary with sample counts for positive and negative classes across trainings (
        Training on all data, Training on train split, Evaluation on test split).
    :param str dsc:
        Description for the data being evaluated.
    :param list clas_names:
        List of class names used in performance metrics.

    Returns:
        None
    """

    output_directory = output_directory + '/ML/'
    os.makedirs(output_directory, exist_ok=True)

    draw_kde_plot(pos_class_prob_values_dict, model_type, clf, output_directory, dsc)

    total_rows = []
    for key in pref_measures_dict.keys():
        row = [key]
        row.extend(num_full_pos_neg_class_samples[key])
        # tn, fp, fn, tp, accuracy, specificity, sensitivity
        row.extend(pref_measures_dict[key])
        total_rows.append(row)

    # Define column names
    columns = ['Process Type', 'Total number of samples', clas_names[0], clas_names[1], 'TN', 'FP', 'FN', 'TP',
               'Accuracy', 'Specificity', 'Sensitivity']

    # Convert to DataFrame
    pref_df = pd.DataFrame(total_rows, columns=columns)

    pref_df.to_csv(output_directory + '/' + model_type + '.' + clf + '.performance.csv', index=False)

    with open(output_directory + '/' + model_type + '.' + clf +
              '.feature_weights_training_all_data.log.txt', "w") as file:
        # Write the outputs to the file
        if feature_importance_df.shape[0] > 0:
            file.write(f"Feature Weights based on model training on all data: {feature_importance_df}\n")


def make_full_data(x_dat, y_dat):

    """
    Combines feature data and target labels into a single DataFrame.

    :param pandas.DataFrame x_dat:
        DataFrame containing feature data.
    :param pandas.Series y_dat:
        Series containing target labels.

    :returns:
        pandas.DataFrame with an added 'real_label' column containing target labels.
    """

    dat = x_dat.copy()
    dat['real_label'] = y_dat.copy()

    return dat


def extract_positive_class_probability_values(pipeline, x_dat, dat):

    """
    Extracts the positive class probability values from a trained model pipeline.

    :param sklearn.pipeline.Pipeline pipeline:
        Trained machine learning pipeline.
    :param pandas.DataFrame x_dat:
        DataFrame containing feature data.
    :param pandas.DataFrame dat:
        DataFrame containing additional data, including real labels.

    :returns:
        numpy.ndarray containing the positive class probability values for the positive class.
    """

    y_prob_train = pipeline.predict_proba(x_dat)[:, 1]

    # Convert the numpy array to a DataFrame
    y_predicted_train_df = pd.DataFrame(y_prob_train, index=x_dat.index, columns=['probability'])

    dat_included_prob = pd.concat([dat, y_predicted_train_df], axis=1)
    dat_class_positive = dat_included_prob.loc[dat_included_prob['real_label'] == 'positive']
    pos_class_prob_values = dat_class_positive['probability'].values

    return pos_class_prob_values


def performance_calculation(y_dat, y_predicted):

    """
    Calculates performance metrics including accuracy, specificity, and sensitivity.

    :param numpy.ndarray y_dat:
        True labels.
    :param numpy.ndarray y_predicted:
        Predicted labels.

    :returns:
        List of performance metrics [tn, fp, fn, tp, accuracy, specificity, sensitivity].
    """

    # Calculate accuracy
    accuracy = accuracy_score(y_dat, y_predicted)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_dat, y_predicted)

    # Calculate specificity and sensitivity from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return [tn, fp, fn, tp, accuracy, specificity, sensitivity]


def numb_positive_negative_class_samples(df):

    """
    Counts the number of positive and negative samples in a DataFrame.

    :param pandas.DataFrame df:
        DataFrame containing a 'real_label' column with class labels.

    :returns:
        Tuple (n_negative, n_positive) with counts of negative and positive samples.
    """

    n_positive = df.loc[df['real_label'] == 'positive'].shape[0]
    n_negative = df.loc[df['real_label'] == 'negative'].shape[0]

    return n_negative, n_positive


def extract_feature_importance(pipeline):

    """
    Extracts feature importance or weights from a trained machine learning pipeline.

    :param sklearn.pipeline.Pipeline pipeline:
        Trained machine learning pipeline.

    :returns:
        pandas.DataFrame with columns 'Feature' and 'Weight', representing feature names
        and their corresponding weights/importance. Returns an empty DataFrame if the
        classifier does not support feature importance.
    """

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

    return feature_importance_df


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
        Descriptive name of model (division_vs_continuity, continuity, division)
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

    # Create a ColumnTransformer to apply StandardScaler to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'
    )

    # Define the pipeline
    full_dat_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_dict[clf])
    ])

    pref_report_pipeline = copy.deepcopy(full_dat_pipeline)

    x_train = x_dat
    y_train = y_dat

    # using for performance report on independent test set
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        x_dat, y_dat, test_size=0.2, stratify=y_dat, random_state=42
    )

    full_dat = make_full_data(x_train, y_train)
    train_split_dat = make_full_data(x_train_split, y_train_split)
    test_split_dat = make_full_data(x_test_split, y_test_split)

    n_negative_full_dat, n_positive_full_dat = numb_positive_negative_class_samples(full_dat)
    n_negative_train_split, n_positive_train_split = numb_positive_negative_class_samples(train_split_dat)
    n_negative_test_split, n_positive_test_split = numb_positive_negative_class_samples(test_split_dat)

    # Train the Logistic Regression model on the scaled data
    # Fit the pipeline on the training data
    full_dat_pipeline.fit(x_train, y_train)
    pref_report_pipeline.fit(x_train_split, y_train_split)

    # Predict on the test set
    y_predicted_train = full_dat_pipeline.predict(x_train)

    y_predicted_train_split = pref_report_pipeline.predict(x_train_split)
    y_predicted_test_split = pref_report_pipeline.predict(x_test_split)

    full_data_pos_class_prob_values = extract_positive_class_probability_values(full_dat_pipeline, x_train, full_dat)
    train_split_dat_data_pos_class_prob_values = \
        extract_positive_class_probability_values(pref_report_pipeline, x_train_split, train_split_dat)
    test_split_dat_pos_class_prob_values = \
        extract_positive_class_probability_values(pref_report_pipeline, x_test_split, test_split_dat)

    # calculate performance measures
    # tn, fp, fn, tp, accuracy, specificity, sensitivity
    pref_measures_full_dat = performance_calculation(y_train, y_predicted_train)
    pref_measures_train_split = performance_calculation(y_train_split, y_predicted_train_split)
    pref_measures_test_split = performance_calculation(y_test_split, y_predicted_test_split)

    num_full_pos_neg_class_samples = \
        {'Training on all data': [x_train.shape[0], n_positive_full_dat, n_negative_full_dat],
         'Training on train split': [x_train_split.shape[0], n_positive_train_split, n_negative_train_split],
         'Evaluation on test split': [x_test_split.shape[0], n_positive_test_split, n_negative_test_split]}

    pos_class_prob_values_dict = {'Training on all data': full_data_pos_class_prob_values,
                                  'Training on train split': train_split_dat_data_pos_class_prob_values,
                                  'Evaluation on test split': test_split_dat_pos_class_prob_values}

    pref_measures_dict = {'Training on all data': pref_measures_full_dat,
                          'Training on train split': pref_measures_train_split,
                          'Evaluation on test split': pref_measures_test_split}

    # Get the weights of features
    feature_importance_df = extract_feature_importance(full_dat_pipeline)

    # Report results
    if model_type != 'division_vs_continuity':
        dsc = 'Highly Likely Correct Links'
        clas_names = ['Number of highly likely correct links', 'Number of highly likely incorrect links']
    else:
        dsc = 'Highly Likely Continuity Links'
        clas_names = ['Number of continuity links', 'Number of division links']
    log_model_performance(output_directory, model_type, pos_class_prob_values_dict, pref_measures_dict,
                          feature_importance_df, clf, num_full_pos_neg_class_samples, dsc, clas_names)

    return full_dat_pipeline
