import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from framingham_score import *
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


def preprocess_data(df, handle_missings='median', add_sample=None):
    if add_sample is not None:
        sample_df = pd.DataFrame([add_sample+[1]], columns=df.columns)
        sample_df = preprocess_data(sample_df,add_sample=None)

    categorical_columns = ['is_smoking', "BPMeds", 'prevalentStroke', 'prevalentHyp', 'diabetes', 'sex', 'education']
    df.loc[df['is_smoking'] == "YES", 'is_smoking'] = 1
    df.loc[df['is_smoking'] == "NO", 'is_smoking'] = 0
    df.loc[df['sex'] == "M", 'sex'] = 1
    df.loc[df['sex'] == "F", 'sex'] = 0
    df["is_smoking"] = df["is_smoking"].astype(int)
    df["sex"] = df["sex"].astype(int)

    if handle_missings == 'median':
        for col in df.columns:
            if col not in categorical_columns:
                mean_col = df[col].median()
                df[col] = df[col].fillna(mean_col)
            else:
                mean_col = np.argmax(df[col])
                df[col] = df[col].fillna(mean_col)
    elif handle_missings == 'knn':
        for col in df.columns:
            if col not in categorical_columns:
                mean_col = df[col].median()
                df[col] = df[col].fillna(mean_col)

        imputator = KNNImputer(n_neighbors=5)
        df = pd.DataFrame(imputator.fit_transform(df), columns=df.columns)

    # Column to see if patient was in Hypertension during the sampling.
    df.insert(len(df.columns) - 1, 'isHyp', value=0)
    df.loc[(df['sysBP'] >= 130) | (df['diaBP'] >= 80), 'isHyp'] = 1

    # creating categorical column for packs of cigarettes.
    df.insert(len(df.columns) - 1, 'packsOfCigs', value=0)
    df.loc[df['cigsPerDay'] > 0, 'packsOfCigs'] = 1
    df.loc[df['cigsPerDay'] >= 10, 'packsOfCigs'] = 2
    df.loc[df['cigsPerDay'] >= 20, 'packsOfCigs'] = 3
    df.loc[df['cigsPerDay'] >= 30, 'packsOfCigs'] = 4

    # creating categorical column for glucose levels.
    df.insert(len(df.columns) - 1, 'glucose_level', value=0)
    df.loc[df['glucose'] > 2.6 * 18, 'glucose_level'] = 1
    df.loc[df['glucose'] > 4.7 * 18, 'glucose_level'] = 2
    df.loc[df['glucose'] > 6.3 * 18, 'glucose_level'] = 3
    df.loc[df['glucose'] > 8.5 * 18, 'glucose_level'] = 4

    # Score2 diabetes score (Framingham Risk Score)
    df.insert(len(df.columns) - 1, 'diabetes_score2', value=0)
    df['diabetes_score2'] = df.apply(lambda x: calculate_framingham_score(
        x['age'], x['sex'], x['is_smoking'], x['cigsPerDay'], x['BPMeds'], x['prevalentStroke'], x['prevalentHyp'],
        x['diabetes'],
        x['totChol'], x['sysBP'], x['diaBP'], x['BMI'], x['heartRate'], x['glucose'], x['education']), axis=1)
    if add_sample is not None:
        return df.set_index('id'), sample_df
    else:
        return df.set_index('id')


def remove_outliers_iqr(df, columns, threshold=1.5):
    # Calculate the IQR for each column
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds to identify outliers
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Filter rows without outliers
    mask = ~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)
    return df[mask]


def predict_sample(sample):
    """
    Predict one list of values.
    :param sample: list of values.
    :return:
    """
    # Load Data
    data_dir = r"C:\Users\soldier109\PycharmProjects\Moshal Medicine Hackaton 2023\Datasets\Cardiovascular Study Dataset"
    # original_cols = pd.read_csv(data_dir + "\\train.csv").columns[:-1]
    trainset,sample= preprocess_data(pd.read_csv(data_dir + "\\train.csv"), handle_missings='median',add_sample=sample)
    # sample = preprocess_data(pd.DataFrame([sample], columns=original_cols, index=[0]))
    X = trainset.drop(columns=['TenYearCHD']).copy()
    y = trainset['TenYearCHD'].copy()
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(X, y)

    rf_classifier = RandomForestClassifier(max_depth=25,
                                           n_estimators=300)
    X_sample = sample.drop(columns=['TenYearCHD'])
    rf_classifier.fit(X_sm, y_sm)
    y_pred = rf_classifier.predict(X_sample)[0]
    y_pred_prob = rf_classifier.predict_proba(X_sample)[0]
    return y_pred, y_pred_prob