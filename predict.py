from flask import jsonify, request
from flask.views import MethodView
import joblib
import pandas as pd
from sklearn.calibration import LabelEncoder
import torch

from classification import LogisticRegression, make_predictions as make_predictions_classification, classification_dataset_predictions
from regression import LinearRegressionModel, make_predictions as make_predictions_regression


def getFieldNames(X):
    field_names = X.columns.tolist()
    field_types = {}

    for field in field_names:
        if X[field].dtype == 'object':
            field_types[field] = 'categorical'
        else:
            field_types[field] = 'numerical'

    categorical_values = {}

    for field in field_names:
        if field_types[field] == 'categorical':
            unique_values = X[field].unique().tolist()
            categorical_values[field] = unique_values
        else:
            categorical_values[field] = []

    response_data = {
        'fields': [
            {'name': field, 'type': field_types[field], 'options':  categorical_values[field]} for field in field_names
        ]
    }

    return response_data


class DatasetClassificationGetPredictFields(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset + '.csv')

        X = data.drop(columns=['class'])

        response_data = getFieldNames(X)

        return jsonify(response_data)


class DatasetRegressionGetPredictFields(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset = req_data.get('dataset')

        data = pd.read_csv('datasets/regression/' + dataset + '.csv')

        X = data.iloc[:, :-1]

        response_data = getFieldNames(X)

        return jsonify(response_data)


class DatasetClassificationGetPredict(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset = req_data.get('dataset')
        latest_params = req_data.get('latest_params')
        input = req_data.get('input')

        data = pd.read_csv('datasets/classification/' + dataset + '.csv')

        dimension = data.shape[1] - 1

        model = LogisticRegression(dimension, 1)

        w = torch.tensor(latest_params['w']).reshape(1, -1)
        b = torch.tensor([latest_params['b']]).reshape(-1)
        model.linear.weight.data = w
        model.linear.bias.data = b

        input = [float(value) for value in input]
        new_data = torch.tensor(input).type(torch.FloatTensor)

        with torch.no_grad():
            prediction = make_predictions_classification(model, new_data)

        return jsonify({'prediction': prediction[0], 'description': classification_dataset_predictions[dataset][int(prediction[0])]})


class GeneratedDataClassificationPredict(MethodView):
    def post(self):
        req_data = request.get_json()

        latest_params = req_data.get('latest_params')
        x1 = float(req_data.get('x1'))
        x2 = float(req_data.get('x2'))

        model = LogisticRegression(2, 1)

        w = torch.tensor(latest_params['w']).reshape(1, -1)
        b = torch.tensor([latest_params['b']]).reshape(-1)
        model.linear.weight.data = w
        model.linear.bias.data = b

        new_data = torch.tensor([x1, x2]).type(torch.FloatTensor)

        with torch.no_grad():
            prediction = make_predictions_classification(model, new_data)

        return jsonify({'prediction': prediction})


class DatasetRegressionGetPredict(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset = req_data.get('dataset')
        latest_params = req_data.get('latest_params')
        input = req_data.get('input')

        data = pd.read_csv('datasets/regression/' + dataset + '.csv')

        dimension = data.shape[1] - 1

        model = LinearRegressionModel(dimension, 1)

        w = torch.tensor(latest_params['w']).reshape(1, -1)
        b = torch.tensor([latest_params['b']]).reshape(-1)
        model.linear.weight.data = w
        model.linear.bias.data = b

        new_data = []

        for i, value in enumerate(input):
            column = data.columns[i]
            if data[column].dtype == 'object':
                label_encoder_path = 'encoders/label_encoders_' + \
                    dataset + '.pkl'  # Path to the label encoder file
                label_encoders = joblib.load(label_encoder_path)
                label_encoder = label_encoders[column]
                encoded_value = label_encoder.transform([value])[0]
                new_data.append(encoded_value)
            else:
                new_data.append(float(value))

        new_data = torch.tensor(new_data).reshape(1, -1)

        with torch.no_grad():
            prediction = make_predictions_regression(model, new_data)

        return jsonify({'prediction': prediction[0], 'outcome': data.iloc[:, -1].name})
