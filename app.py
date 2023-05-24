import os
from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch

from generateClassificationData import generate_separable_data, generate_non_separable_data, get_params
from classification import LogisticRegression, detect_outliers, make_predictions, datasets

app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80)
    app.run(debug=True)


class HelloWorld(MethodView):
    def get(self):
        return jsonify(welcome="hello")


app.add_url_rule("/test", view_func=HelloWorld.as_view("hello_world"))


class SeparableDataClassification(MethodView):
    def post(self):
        req_data = request.get_json()
        n_samples = int(req_data.get('n_samples', '1000'))
        centers = int(req_data.get('classes', '2'))
        train_size = float(req_data.get('train_size', '0.5'))

        result = generate_separable_data(n_samples, centers, train_size)

        return jsonify(result)


app.add_url_rule('/classification/separable_data',
                 view_func=SeparableDataClassification.as_view('separable_data'))


class NonSeparableDataClassification(MethodView):
    def post(self):
        req_data = request.get_json()
        n_samples = int(req_data.get('n_samples', '1000'))
        centers = int(req_data.get('classes', '2'))
        train_size = float(req_data.get('train_size', '0.5'))

        result = generate_non_separable_data(n_samples, centers, train_size)

        return jsonify(result)


app.add_url_rule('/classification/non_separable_data',
                 view_func=NonSeparableDataClassification.as_view('non_separable_data'))


class ClassificationTrain(MethodView):
    def post(self):
        req_data = request.get_json()

        train_data = (req_data.get('data'))
        dataset = req_data.get('dataset')
        max_iter = int(req_data.get('max_iter'))
        learning_rate = float(req_data.get('learning_rate'))
        optimizer_name = req_data.get('optimizer')
        criterion_name = req_data.get('criterion')

        if optimizer_name == '' or optimizer_name == None:
            return jsonify({'error': 'Optimizer name cannot be empty.'}), 400

        if criterion_name == '' or criterion_name == None:
            return jsonify({'error': 'Criterion name cannot be empty.'}), 400

        if train_data is not None and dataset is None:
            x = np.array([(d['x1'], d['x2']) for d in train_data])
            y = np.array([d['color'] for d in train_data])

        elif train_data is not None and dataset is not None:
            x = []
            y = []

            for d in train_data:
                x_values = [v for k, v in d.items() if k != 'target']
                x.append(x_values)
                y.append(d['target'])
            x = np.array(x)
            y = np.array(y)

        dimension = x[0].size
        model = LogisticRegression(dimension, 1)

        optimizer = getattr(torch.optim, optimizer_name)(
            model.parameters(), lr=learning_rate)
        criterion = getattr(torch.nn, criterion_name)()

        result = get_params(max_iter, model, optimizer,
                            criterion, x, y, dimension)

        return jsonify(result)


app.add_url_rule('/classification/train',
                 view_func=ClassificationTrain.as_view('classification_train'))


class GeneratedDataClassificationTest(MethodView):
    def post(self):
        req_data = request.get_json()

        test_data = req_data.get('test_data')
        train_data = req_data.get('train_data')
        line_params = req_data.get('line_params')

        model = LogisticRegression(2, 1)

        w1 = line_params[-1]['w1']
        w2 = line_params[-1]['w2']
        b = line_params[-1]['b']
        model.linear.weight.data = torch.Tensor([[w1, w2]])
        model.linear.bias.data = torch.Tensor([b])

        x_test = np.array([(d['x1'], d['x2']) for d in test_data])
        y_test = np.array([d['color'] for d in test_data])
        x_train = np.array([(d['x1'], d['x2']) for d in train_data])
        y_train = np.array([d['color'] for d in train_data])

        predictions_test = make_predictions(model, x_test)
        predictions_train = make_predictions(model, x_train)

        accuracy_train = accuracy_score(y_train, predictions_train)
        accuracy_test = accuracy_score(y_test, predictions_test)

        cm_test = confusion_matrix(y_test, predictions_test)
        f1_test = f1_score(y_test, predictions_test)

        cm_train = confusion_matrix(y_train, predictions_train)
        f1_train = f1_score(y_train, predictions_train)

        result = {'confusion_matrix_test': cm_test.tolist(),
                  'f1_score_test': f1_test * 100,
                  'confusion_matrix_train': cm_train.tolist(),
                  'f1_score_train': f1_train * 100,
                  'accuracy_train': accuracy_train * 100, 'accuracy_test': accuracy_test * 100}
        return jsonify(result)


app.add_url_rule('/classification/generated/test',
                 view_func=GeneratedDataClassificationTest.as_view('separable_data_test'))


class GeneratedDataClassificationPredict(MethodView):
    def post(self):
        req_data = request.get_json()

        line_params = req_data.get('line_params')
        x1 = float(req_data.get('x1'))
        x2 = float(req_data.get('x2'))

        model = LogisticRegression(2, 1)

        w1 = line_params[-1]['w1']
        w2 = line_params[-1]['w2']
        b = line_params[-1]['b']
        model.linear.weight.data = torch.Tensor([[w1, w2]])
        model.linear.bias.data = torch.Tensor([b])

        new_data = torch.tensor([x1, x2]).type(torch.FloatTensor)

        with torch.no_grad():
            prediction = make_predictions(model, new_data)

        return jsonify({'prediction': prediction})


app.add_url_rule('/classification/generated/predict',
                 view_func=GeneratedDataClassificationPredict.as_view('separable_data_predict'))


class ClassificationDataset(MethodView):
    def get(self):
        return jsonify(datasets)


app.add_url_rule('/classification/datasets',
                 view_func=ClassificationDataset.as_view('classification_datasets'))


class ClassificationDatasetStatisticalAnalysis(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        dataset_path = 'datasets/classification/' + dataset_name + '.csv'
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 400

        for dataset in datasets:
            if dataset['value'] == dataset_name:
                label = dataset['label']

        data = pd.read_csv(dataset_path)

        data_stats = []
        columns = data.columns

        for column in columns:
            # Check if column data type is float or integer
            is_numerical = data[column].dtype.kind in 'fi'
            stats = {
                'column': column,
                'null': int(data[column].isnull().sum()),
                'unique': int(data[column].nunique()),
                'max': data[column].max().item(),
                'min': data[column].min().item(),
                'std': data[column].std().item(),
                'mean': data[column].mean().item(),
                'median': data[column].median().item(),
                'is_numerical': is_numerical,
            }
            data_stats.append(stats)

        return jsonify({'data_stats': data_stats, 'label': label})


app.add_url_rule('/classification/dataset/statistical-analysis',
                 view_func=ClassificationDatasetStatisticalAnalysis.as_view('dataset_statistical_analysis'))


class ClassificationBoxPlot(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        numerical_columns = data.select_dtypes(
            include=['float64', 'int64']).columns
        box_plot_data = []

        for column in numerical_columns:
            box_plot_values = [
                data[column].min().item(),
                data[column].quantile(0.25).item(),
                data[column].median().item(),
                data[column].quantile(0.75).item(),
                data[column].max().item()
            ]
            box_plot_entry = {
                'x': column,
                'y': box_plot_values
            }
            box_plot_data.append(box_plot_entry)

        return jsonify(box_plot_data)


app.add_url_rule('/classification/dataset/box-plot',
                 view_func=ClassificationBoxPlot.as_view('box_plot_data'))


class ClassificationHistogram(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        numerical_columns = data.select_dtypes(
            include=['float64', 'int64']).columns
        histogram_data = []

        for column in numerical_columns:
            column_data = data[column].dropna()
            iqr = np.percentile(column_data, 75) - \
                np.percentile(column_data, 25)

            bin_width = 2 * iqr * (len(column_data) ** (-1/3))

            num_bins = int(
                (column_data.max() - column_data.min()) / bin_width)

            histogram_values, histogram_bins = np.histogram(
                column_data, bins=num_bins)

            histogram_data_entry = {
                'column': column,
                'x': histogram_bins.tolist(),
                'y': histogram_values.tolist(),
            }
            histogram_data.append(histogram_data_entry)

        return jsonify(histogram_data)


app.add_url_rule('/classification/dataset/histogram',
                 view_func=ClassificationHistogram.as_view('histogram'))


class ClassificationDatasetSets(MethodView):
    def post(self):
        req_data = request.get_json()
        train_size = float(req_data.get('train_size', '0.5'))
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        X = data.drop(columns=['class'])
        y = data['class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size)

        train_data = [dict(x, target=c) for x, c in zip(
            X_train.to_dict(orient='records'), y_train)]
        test_data = [dict(x, target=c) for x, c in zip(
            X_test.to_dict(orient='records'), y_test)]

        return jsonify({
            'train_data': train_data,
            'test_data': test_data
        })


app.add_url_rule('/classification/dataset/sets',
                 view_func=ClassificationDatasetSets.as_view('dataset_sets'))
