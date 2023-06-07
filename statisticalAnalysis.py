import os
from flask import jsonify, request
from flask.views import MethodView

import pandas as pd
import numpy as np

from classification import classification_datasets
from regression import regression_datasets

# TODO: Return boolean if model.pt exists (to show button to skip to predict immediately)


def get_data_stats(data):
    data_stats = []
    num_data_stats = []
    columns = data.columns

    for column in columns:
        # Check if column data type is float or integer
        is_numerical = data[column].dtype.kind in 'fi'
        stats = {
            'column': column,
            'null': int(data[column].isnull().sum()),
            'unique': int(data[column].nunique()),
            'is_numerical': is_numerical,
        }
        data_stats.append(stats)

        if is_numerical:
            num_stats = {
                'column': column,
                'max': data[column].max().item(),
                'min': data[column].min().item(),
                'std': data[column].std().item(),
                'mean': data[column].mean().item(),
                'median': data[column].median().item(),
            }
            num_data_stats.append(num_stats)

    return data_stats, num_data_stats


class ClassificationDatasetStatisticalAnalysis(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        dataset_path = 'datasets/classification/' + dataset_name + '.csv'
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 400

        for dataset in classification_datasets:
            if dataset['value'] == dataset_name:
                label = dataset['label']

        data = pd.read_csv(dataset_path)

        data_size = {'count': data.shape[0], 'dimension': data.shape[1] - 1}

        data_stats, num_data_stats = get_data_stats(data)

        model_path = 'models/classification/' + dataset_name + '.pt'
        if os.path.exists(model_path):
            model_exists = True
        else:
            model_exists = False

        return jsonify({'data_stats': data_stats, 'num_data_stats': num_data_stats, 'label': label, 'data_size': data_size, 'model_exists': model_exists})


class RegressionDatasetStatisticalAnalysis(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        dataset_path = 'datasets/regression/' + dataset_name + '.csv'
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 400

        for dataset in regression_datasets:
            if dataset['value'] == dataset_name:
                label = dataset['label']

        data = pd.read_csv(dataset_path)

        data_size = {'count': data.shape[0], 'dimension': data.shape[1] - 1}

        data_stats, num_data_stats = get_data_stats(data)

        model_path = 'models/regression/' + dataset_name + '.pt'
        if os.path.exists(model_path):
            model_exists = True
        else:
            model_exists = False

        return jsonify({'data_stats': data_stats, 'num_data_stats': num_data_stats, 'label': label, 'data_size': data_size, 'model_exists': model_exists})


def getBoxPlotData(data):
    numerical_columns = data.select_dtypes(
        include=['float64', 'int64']).columns
    box_plot_data = []

    for column in numerical_columns:
        column_data = data[column]
        q1 = column_data.quantile(0.25)
        q3 = column_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Calculate outliers
        outliers = column_data[(column_data < lower_bound) | (
            column_data > upper_bound)]

        # Exclude outliers from the box plot data
        filtered_data = column_data[(column_data >= lower_bound) & (
            column_data <= upper_bound)]

        box_plot_values = [
            filtered_data.min().item(),
            filtered_data.quantile(0.25).item(),
            filtered_data.median().item(),
            filtered_data.quantile(0.75).item(),
            filtered_data.max().item()
        ]
        box_plot_entry = {
            'x': column,
            'y': box_plot_values,
            'outliers': outliers.tolist()  # Convert outliers to a list
        }
        box_plot_data.append(box_plot_entry)

    return box_plot_data


class ClassificationBoxPlot(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        box_plot_data = getBoxPlotData(data)
        return jsonify(box_plot_data)


class RegressionBoxPlot(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/regression/' + dataset_name + '.csv')

        box_plot_data = getBoxPlotData(data)
        return jsonify(box_plot_data)


def getHistogramData(data):
    numerical_columns = data.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    histogram_data = []

    for column in numerical_columns:
        column_data = data[column].dropna()
        iqr = np.percentile(column_data, 75) - \
            np.percentile(column_data, 25)

        bin_width = 2 * iqr * (len(column_data) ** (-1/3))

        if bin_width > 0:
            num_bins = int((column_data.max() - column_data.min()) / bin_width)
        else:
            num_bins = 1

        histogram_values, histogram_bins = np.histogram(
            column_data, bins=num_bins)

        histogram_data_entry = {
            'column': column,
            'x': histogram_bins.tolist(),
            'y': histogram_values.tolist(),
        }
        histogram_data.append(histogram_data_entry)

    return histogram_data


class ClassificationHistogram(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        histogram_data = getHistogramData(data)
        return jsonify(histogram_data)


class RegressionHistogram(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/regression/' + dataset_name + '.csv')

        histogram_data = getHistogramData(data)
        return jsonify(histogram_data)


def getCountPlotData(data):
    categorical_columns = data.select_dtypes(
        include=['object']).columns.tolist()
    count_plot_data = []

    for column in categorical_columns:
        column_data = data[column].dropna()
        counts = column_data.value_counts().reset_index()

        count_plot_data_entry = {
            'column': column,
            'x': counts['index'].tolist(),
            'y': counts[column].tolist(),
        }
        count_plot_data.append(count_plot_data_entry)

    return count_plot_data


class ClassificationCountPlot(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        count_plot_data = getCountPlotData(data)
        return jsonify(count_plot_data)


class RegressionCountPlot(MethodView):
    def post(self):
        req_data = request.get_json()
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/regression/' + dataset_name + '.csv')

        count_plot_data = getCountPlotData(data)
        return jsonify(count_plot_data)
