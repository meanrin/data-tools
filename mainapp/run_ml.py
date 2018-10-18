import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, precision_recall_curve, confusion_matrix
from .filehandler import read_file

model_dictionary = {
    'Linear Regression': LinearRegression,

    'Logistic regression': LogisticRegression,
}

def run_ml(filename, columns_description, model):
    dt = read_file(filename).data

    numeric_columns = [
        c for c in dt.columns if
        c in columns_description and
        columns_description[c][-1] != 'exclude' and
        dt[c].dtype != 'O' and
        columns_description[c][0] == 'false'
    ]
    categorical_columns = [
        c for c in dt.columns if
        c in columns_description and
        columns_description[c][-1] != 'exclude' and
        (dt[c].dtype == 'O' or
         columns_description[c][0] == 'true')
    ]

    X = np.zeros((dt.shape[0], 0))

    task = None
    y_unique = None

    for c in numeric_columns:
        _, nulls, scale, bins, type = columns_description[c]
        bins = 0 if len(bins) == 0 else int(bins)

        values = dt[c].copy()

        if nulls == 'mean':
            values = values.fillna(values.mean())
        elif nulls == 'median':
            values = values.fillna(values.median())
        elif nulls == 'mode':
            values = values.fillna(values.mode()[0])

        if scale == 'norm':
            values = (values-values.mean())/values.std()
        elif scale == 'stan':
            values = values/values.max()
        elif scale == 'bins':
            pass
        elif scale == 'log':
            values = np.log(values)

        if type == 'input':
            X = np.hstack((X, values[:,np.newaxis]))
        elif type == 'target':
            task = 'reg'
            Y = values


    for c in categorical_columns:
        type = columns_description[c][-1]

        unique = list(set(dt[c]))

        if type == 'input':
            if len(unique) == 2:
                X = np.hstack((X, np.array([unique.index(x) for x in dt[c]])[:, np.newaxis]))
            else:
                X = np.hstack((X, pd.get_dummies(dt[c], c)))
        elif type == 'target':
            print(c)
            if len(unique) == 2:
                task = 'bin'
            else:
                task = 'cat'
            Y = np.array([unique.index(x) for x in dt[c]])
            y_unique = unique

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, shuffle=True)

    model = model_dictionary[model]()
    model.fit(X_train, Y_train)

    if task == 'reg':
        predict = model.predict(X_test)

        r2 = r2_score(Y_test, predict)
        mse = np.mean((predict - Y_test) ** 2)
        scores = {'r2': r2, 'mse': mse}
    elif task == 'bin':
        predict_proba = model.predict_proba(X_test)[:, 1]

        pr, rc, tr = precision_recall_curve(Y_test, predict_proba)
        scores = {'pr': pr.tolist(), 'rc': rc.tolist(), 'tr': tr.tolist(), 'labels': y_unique}
    elif task == 'cat':
        predict = model.predict(X_test)

        cf = confusion_matrix(Y_test, predict, labels=range(len(y_unique)))
        scores = {'cf': cf.tolist(), 'labels': y_unique}



    ml_result = {
        'task': task,
        'scores': scores
    }

    return ml_result