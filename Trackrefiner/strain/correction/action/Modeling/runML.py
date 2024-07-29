from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import gzip
import pickle
import numpy as np


def write_model(output_directory, stat, model, train_dat, test_dat, accuracy, specificity, sensitivity,
                feature_importance_df, tp, tn, fp, fn, clf):

    output_directory = output_directory + '/ML/'
    os.makedirs(output_directory, exist_ok=True)

    # saving the trained model
    model_file_name = output_directory + stat + '.' + clf + '.model.gz'
    with gzip.open(model_file_name, 'wb') as f:
        pickle.dump(model, f)

    train_dat.to_csv(output_directory + '/' + stat + '.' + clf + '.training.csv', index=False)
    test_dat.to_csv(output_directory + '/' + stat + '.' + clf + '.test.csv', index=False)

    with open(output_directory + '/' + stat + '.' + clf + '.performance.log.txt', "w") as file:
        # Write the outputs to the file
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Specificity: {specificity}\n")
        file.write(f"Sensitivity: {sensitivity}\n")
        file.write(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}\n")
        if feature_importance_df.shape[0] > 0:
            file.write(f"Feature Weights: {feature_importance_df}\n")


def run_ml_model(merged_df, feature_list, columns_to_scale, stat, output_directory, clf, n_cpu):

    # 'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=n_cpu),
    # 'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    #                 'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
    #                 'ExtraTreeClassifier': ExtraTreeClassifier(random_state=42),
    # 'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    # 'Nu-Support Vector Classifier': NuSVC(random_state=42, probability=True)
    # 'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(store_covariance=False),
    clf_dict = {
                'LogisticRegression': LogisticRegression(random_state=42, n_jobs=n_cpu, class_weight='balanced'),
                'GaussianProcessClassifier': GaussianProcessClassifier(random_state=42, n_jobs=n_cpu),
                'C-Support Vector Classifier': SVC(random_state=42, probability=True, class_weight='balanced')}

    # Define features and target
    X = merged_df[feature_list]
    y = merged_df['label']

    # Encode string labels to binary values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Inspect the class mapping
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # print(f"Class Mapping: {class_mapping}")

    # Create a ColumnTransformer to apply StandardScaler to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), columns_to_scale)
        ],
        remainder='passthrough'
    )

    # Define the pipeline with SMOTE
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_dict[clf])
    ])

    # Split the data using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Apply SMOTE to the training data
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    train_dat = X_train.copy()
    train_dat['label'] = y_train.copy()

    test_dat = merged_df.loc[merged_df.index.isin(X_test.index.values)]

    # X_train_scaled = X_train
    # X_test_scaled = X_test

    # Train the Logistic Regression model on the scaled data
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred_test = pipeline.predict(X_test)

    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Convert the numpy array to a DataFrame
    y_pred_train_df = pd.DataFrame(y_prob_train, index=X_train.index, columns=['probability'])
    y_pred_test_df = pd.DataFrame(y_prob_test, index=X_test.index, columns=['probability'])

    train_dat = pd.concat([train_dat, y_pred_train_df], axis=1)
    test_dat = pd.concat([test_dat, y_pred_test_df], axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_test)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Calculate specificity and sensitivity from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
    ppv = tp / (tp + fp)  # Precision
    npv = tn / (tn + fn)

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
    except:
        try:
            # Try to get feature importances for models that have feature_importances_ attribute
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
    write_model(output_directory, stat, model, train_dat, test_dat, accuracy, specificity, sensitivity,
                feature_importance_df, tp, tn, fp, fn, clf)

    return pipeline

