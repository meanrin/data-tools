from django.shortcuts import render

from django import forms

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt,csrf_protect

from .run_ml import run_ml
from . import filehandler

import pandas as pd
import numpy as np

import os


def simple_upload(request):
    filename = None
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        fs.save('storage/{}'.format(myfile.name), myfile)
        filename = myfile.name

    return render(request, 'mainapp/simple_upload.html', {
        'uploaded_files': os.listdir('storage'),
        'new_file': filename
    })

def handle_file(request, name):

    dt = filehandler.read_file(name)

    try:
        numeric_description = dt.data.describe(include=[np.number]).round(4)
        numeric_description = {c:
                                       {
                                           desc:value for desc,value in dict(numeric_description[c]).items()
                                           if not pd.isnull(value)
                                       }
                            for c in numeric_description.columns}
    except:
        numeric_description = {}
    numeric_columns = list(numeric_description.keys())

    try:
        categorical_description = dt.data.describe(include=['O']).round(4)
        categorical_description = {c:
            {
                desc: value for desc, value in dict(categorical_description[c]).items()
                if not pd.isnull(value)
            }
            for c in categorical_description.columns}
    except:
        categorical_description = {}
    cat_columns = list(categorical_description.keys())

    NaN_data = dt.data.fillna("NaN")

    cat_values = {c:dict(NaN_data[c].value_counts()) for c in cat_columns}

    correlation = dt.data[numeric_columns].corr().round(4)

    return render(request, 'mainapp/research.html', {
        'column_names': dt.data.columns,
        'numeric_values': dict(zip(
            numeric_description.keys(),
            np.array(NaN_data[list(numeric_description.keys())]).T.tolist()
        )),
        'cat_values': cat_values,
        'correlation': dict(zip(correlation.columns, np.array(correlation).tolist())),

        'column_description': {**numeric_description, **categorical_description},
    })


def prepare_ml(request, name):

    data = filehandler.read_file(name).data

    column_type = {}

    for c in data.columns:
        if sum(data[c].isnull()) == len(data[c]):
            continue

        column_type[c] = 'number' if data[c].dtype != 'object' else \
            'binary' if len(set(data[c])) == 2 else \
                'binary_nan' if len(set(data[c])) == 3 and sum(data[c].isnull()) > 0 else \
                    'categorical'

    return render(request, 'mainapp/prepare_ml.html', {
        'column_type': column_type,
        'regression_models': ['Linear Regression', ],#'Gradient Boosting', 'Support Vector Regression'],
        'classification_models': ['Logistic regression', ]#'Gradient Boosting', 'Random Forest', 'Support Vector Classifier'],
    })

@csrf_exempt
def submit(request):
    description = dict(request.POST)

    model = description['__selected_model__'][0]
    filename = description['__filename__'][0]
    description.pop('__selected_model__', None)
    description.pop('__filename__', None)
    description = {k:v[0].split(',') for k,v in description.items()}
    scores = run_ml(filename, description, model)
    return JsonResponse(scores)