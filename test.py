import numpy as np
import pandas as pd
import joblib

# importing necessary files for preprocessing of data
imputer = joblib.load('models/imputer.joblib')
std_scaler = joblib.load('models/std_scaler.joblib')
dumb5_ohe = joblib.load('models/dumb5_ohe.joblib')
dumb31_ohe = joblib.load('models/dumb31_ohe.joblib')
dumb81_ohe = joblib.load('models/dumb81_ohe.joblib')
dumb82_ohe = joblib.load('models/dumb82_ohe.joblib')
var_reduced = joblib.load('models/var_reduced.joblib')
result = joblib.load('models/result.joblib')


def preprocess(raw_test):
    """
    pre-processing of Raw data

    arguments raw_test  = Pandas DataFrame Object with same structure as training data

    returns preprocessed data
    """

    # data cleaning of columns
    raw_test['x12'] = raw_test['x12'].str.replace('$', '')
    raw_test['x12'] = raw_test['x12'].str.replace(',', '')
    raw_test['x12'] = raw_test['x12'].str.replace(')', '')
    raw_test['x12'] = raw_test['x12'].str.replace('(', '-')
    raw_test['x12'] = raw_test['x12'].astype(float)
    raw_test['x63'] = raw_test['x63'].str.replace('%', '')
    raw_test['x63'] = raw_test['x63'].astype(float)

    # dropping columns to one hot encode them later
    test = raw_test.drop(columns=['x5', 'x31', 'x81', 'x82'])

    # replacing nan by mean of columns followed by standard Scaling
    test_std = pd.DataFrame(imputer.transform(test), columns=test.columns)
    test_std = pd.DataFrame(std_scaler.transform(test_std), columns=test.columns)

    # loading one hot encode model -> processing respective column  -> renaming and concating to dataframe
    dumb5 = dumb5_ohe.transform(raw_test['x5'].fillna('NaN').values.reshape(-1, 1))
    test_std = pd.concat([test_std,
                          pd.DataFrame(dumb5,
                                       columns=[f'x5_{i}' for i in dumb5_ohe.categories_[0][1:]])],
                         axis=1, sort=False)

    dumb31 = dumb31_ohe.transform(raw_test['x31'].fillna('NaN').values.reshape(-1, 1))
    test_std = pd.concat([test_std,
                          pd.DataFrame(dumb31,
                                       columns=[f'x31_{i}' for i in dumb31_ohe.categories_[0][1:]])],
                         axis=1, sort=False)

    dumb81 = dumb81_ohe.transform(raw_test['x81'].fillna('NaN').values.reshape(-1, 1))
    test_std = pd.concat([test_std,
                          pd.DataFrame(dumb81,
                                       columns=[f'x81_{i}' for i in dumb81_ohe.categories_[0][1:]])],
                         axis=1, sort=False)

    dumb82 = dumb82_ohe.transform(raw_test['x82'].fillna('NaN').values.reshape(-1, 1))
    test_std = pd.concat([test_std,
                          pd.DataFrame(dumb82,
                                       columns=[f'x82_{i}' for i in dumb82_ohe.categories_[0][1:]])],
                         axis=1, sort=False)

    # using column names generated while training to reduce to 25 most useful columns
    return test_std[var_reduced]


def prediction(data):
    """
    prediction of values given the dataset

    arguments data: of format dictionary or list of dictionaries containing dataset

    return dictionary with class , probability and name of columns'
    """

    # checking for type of data given
    if isinstance(data, dict):
        raw_test = pd.DataFrame(data, index=[0])
    else:
        raw_test = pd.DataFrame(data)

    # pre-processing data
    process = preprocess(raw_test)

    # calculating result using saved model
    y_pred = result.predict(process)

    # returning list dictionaries
    to_be_returned = []

    for i in range(len(y_pred)):
        dic = dict()
        dic['class'] = int(np.where(y_pred[i] > 0.5, 1, 0))  # determining class
        dic['probability'] = float(y_pred[i])  # probability of class
        dic['columns'] = list(var_reduced)  # list of columns used to train
        to_be_returned.append(dic)
    return to_be_returned
