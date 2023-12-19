import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import rce_local
from scipy.stats import sem
import warnings
import tqdm
from framingham_score import *
from sklearn.impute import KNNImputer
import os
from MoshalGUI import blockPrint,enablePrint



def preprocess_data(df, handle_missings='median', add_sample=None):
    if add_sample is not None:
        sample_df = pd.DataFrame([add_sample + [1]], columns=df.columns)
        sample_df = preprocess_data(sample_df, add_sample=None)

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


def run_experiment(df, target_column, norm, dataset_name="", immutable_features=[], positive_features=[],
                   binary_features=[],
                   integers_features=[], coherence_features={}, larger_than_features=[],
                   exclude_models=[], num_instances=20, time_limit=1000,
                   rho=0.05, trees_max_depth=10, trees_n_est=50):
    """
    :param df: dataframe of train data
    :param norm:  'linf' or 'l2'
    :param exclude_models: list of models to skip. models can be: linear, cart, rf, mlp, gbm.
    :param num_instances:  number of points to sample and find CEs for.
    :param time_limit: time limit for optimizer.
    :param rho: rho.
    """
    num_instances = num_instances
    time_limit = time_limit
    rho = rho
    uncertainty_type = norm

    clf_dict = {'rf': [100], 'gbm': [100]}

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = df.iloc[:, :-1]
    y = df[target_column]

    warnings.filterwarnings('ignore')

    num_iterations_dict = {(i, j): [] for i in clf_dict.keys() for j in clf_dict[i]}
    comp_time_dict = {(i, j): [] for i in clf_dict.keys() for j in clf_dict[i]}
    dist_early_stops = {(i, j): [] for i in clf_dict.keys() for j in clf_dict[i]}
    early_stops_iter = {(i, j): [] for i in clf_dict.keys() for j in clf_dict[i]}

    res = []
    solutions = []
    for clf_type in clf_dict.keys():
        if clf_type in exclude_models:
            continue
        for param in clf_dict[clf_type]:
            if clf_type == 'rf':
                clf = RandomForestClassifier(max_depth=trees_max_depth, random_state=0, n_estimators=trees_n_est).fit(X,
                                                                                                                      y)
            elif clf_type == 'gbm':
                clf = GradientBoostingClassifier(n_estimators=trees_n_est, learning_rate=0.1, max_depth=trees_max_depth,
                                                 random_state=0).fit(X, y)

            for i in range(num_instances):
                if y.iloc[i] != 1:
                    continue
                print(f'#### Model: {clf_type}\tSpecs: {param}\tSample number: {i + 1}/{num_instances} ####')
                np.random.seed(i)
                u = pd.DataFrame([X.iloc[i, :]])

                original_sample = scaler.inverse_transform(np.array([list(X.iloc[i, :]) + [y.iloc[i]]]))[0]
                solutions.append([i, clf_type, param] + list(original_sample))

                it = True if clf_type != 'linear' else False
                final_model, num_iterations, comp_time, x_, solutions_master_dict = rce_local.generate(clf, X, y,
                                                                                                       './results_%s' % dataset_name,
                                                                                                       clf_type,
                                                                                                       'binary', u,
                                                                                                       list(u.columns),
                                                                                                       binary_features,
                                                                                                       integers_features,
                                                                                                       coherence_features,
                                                                                                       immutable_features,
                                                                                                       larger_than_features,
                                                                                                       positive_features,
                                                                                                       rho,
                                                                                                       unc_type=uncertainty_type,
                                                                                                       iterative=it,
                                                                                                       time_limit=time_limit)

                if x_ is not None:
                    solution_subopt, dist = rce_local.find_maxrad(x_, clf_type, 'results_%s' % dataset_name, x_.columns,
                                                                  binary_features,
                                                                  integers_features, coherence_features,
                                                                  immutable_features, larger_than_features,
                                                                  positive_features, clf.predict(u)[0],
                                                                  uncertainty_type)
                if x_ is None or dist + rho / 100 < rho:
                    best_dist = 0
                    for i in range(len(solutions_master_dict)):
                        x_ = solutions_master_dict[i]['sol']
                        solution_subopt_i, dist_i = rce_local.find_maxrad(x_, clf_type, 'results_%s' % dataset_name,
                                                                          x_.columns, binary_features,
                                                                          integers_features, coherence_features,
                                                                          immutable_features, larger_than_features,
                                                                          positive_features, clf.predict(u)[0],
                                                                          uncertainty_type)
                        if dist_i >= best_dist:
                            best_dist = dist_i

                    print(best_dist)
                    dist_early_stops[(clf_type, param)].append(best_dist)
                    early_stops_iter[(clf_type, param)].append(num_iterations)
                    print(
                        '\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ERROR @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')

                else:
                    num_iterations_dict[(clf_type, param)].append(num_iterations)
                    comp_time_dict[(clf_type, param)].append(comp_time)

                # write results to .txt file
                if x_ is not None:
                    ce_of_sample = scaler.inverse_transform(np.array([x_.values.tolist()[0] + [y.iloc[i]]]))[0]
                    solutions.append([i, clf_type, param] + list(ce_of_sample))
                num_iterations_list = num_iterations_dict[(clf_type, param)]
                comp_time_list = comp_time_dict[(clf_type, param)]
                dist_early_stops_list = dist_early_stops[(clf_type, param)]
                early_stops_iter_list = early_stops_iter[(clf_type, param)]

                res.append([clf_type, param, np.mean(comp_time_list), sem(comp_time_list), np.mean(num_iterations_list),
                            sem(num_iterations_list), len(dist_early_stops_list), np.mean(dist_early_stops_list),
                            sem(dist_early_stops_list), np.mean(early_stops_iter_list), sem(early_stops_iter_list)])

    res_df = pd.DataFrame(res,
                          columns=['Model', 'Specs', 'Comp. time', 'Comp. time (std)', 'iterations', 'iterations (std)',
                                   '# early stops', 'dist of early stops', 'dist of early stops (std)',
                                   'early stops iter', 'early stops iter(std)'])
    solution_df = pd.DataFrame(solutions, columns=['Sample ID', 'Model', 'Specs'] + list(df.columns))
    return res_df, solution_df.drop_duplicates(ignore_index=True)


def provide_plan_of_action(original_sample, ce_of_sample, df_columns):
    diffrence = pd.DataFrame([original_sample - ce_of_sample],columns=df_columns)
    diffrence = diffrence.drop(columns=['Model','Sample ID','Specs']).round(decimals=5)
    return f"{diffrence.iloc[0]}\n==================================================\n"


def run_ce_optimization(sample):
    # Load Data
    data_dir = r"C:\Users\soldier109\PycharmProjects\Moshal Medicine Hackaton 2023\Datasets\Cardiovascular Study Dataset"
    trainset, sample = preprocess_data(pd.read_csv(data_dir + "\\train.csv"), handle_missings='median',
                                       add_sample=sample)

    immutable_features = ['sex', 'age', 'education','prevalentStroke','prevalentHyp','diabetes']
    positive_features = ["cigsPerDay", "totChol", 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', "packsOfCigs",
                         'glucose_level']
    binary_features = ['is_smoking', "BPMeds", 'isHyp']
    integers_features = []
    larger_than_features = []
    # exclude = []
    exclude = ['rf']
    norm = "l2"
    target_column = 'TenYearCHD'
    std_res = blockPrint()
    res, sol = run_experiment(trainset, target_column, dataset_name="CHD_prediction_" + norm, norm=norm,
                              exclude_models=exclude,
                              immutable_features=immutable_features,
                              positive_features=positive_features,
                              binary_features=binary_features, integers_features=integers_features,
                              larger_than_features=larger_than_features,
                              num_instances=1, time_limit=100, rho=0.05, trees_max_depth=10, trees_n_est=5)
    enablePrint(std_res)
    return provide_plan_of_action(trainset.iloc[0], sol.iloc[1], sol.columns)
