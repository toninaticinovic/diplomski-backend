from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_cors import CORS
import pandas as pd

import torch
from datasets import ClassificationDataset, RegressionDataset

from generateData import NonSeparableDataClassification, SeparableDataClassification

from classification import ClassificationTrain, DataClassificationTest
from predict import DatasetClassificationGetPredict, DatasetClassificationGetPredictFields, DatasetRegressionGetPredict, DatasetRegressionGetPredictFields, GeneratedDataClassificationPredict
from regression import DataRegressionTest, RegressionTrain
from sets import ClassificationDatasetSets, RegressionDatasetSets
from statisticalAnalysis import ClassificationBoxPlot, ClassificationCountPlot, ClassificationDatasetStatisticalAnalysis, ClassificationHistogram, RegressionBoxPlot, RegressionCountPlot, RegressionDatasetStatisticalAnalysis, RegressionHistogram

app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80)
    app.run(debug=True)


class HelloWorld(MethodView):
    def get(self):
        return jsonify(welcome="hello")


app.add_url_rule("/test", view_func=HelloWorld.as_view("hello_world"))

app.add_url_rule('/classification/separable_data',
                 view_func=SeparableDataClassification.as_view('separable_data'))

app.add_url_rule('/classification/non_separable_data',
                 view_func=NonSeparableDataClassification.as_view('non_separable_data'))

app.add_url_rule('/classification/train',
                 view_func=ClassificationTrain.as_view('classification_train'))

app.add_url_rule('/classification/test',
                 view_func=DataClassificationTest.as_view('separable_data_test'))

app.add_url_rule('/classification/dataset/sets',
                 view_func=ClassificationDatasetSets.as_view('classification_dataset_sets'))

app.add_url_rule('/classification/dataset/predict-fields',
                 view_func=DatasetClassificationGetPredictFields.as_view('classification_dataset_predict_fields'))

app.add_url_rule('/classification/generated/predict',
                 view_func=GeneratedDataClassificationPredict.as_view('separable_data_predict'))

app.add_url_rule('/classification/datasets',
                 view_func=ClassificationDataset.as_view('classification_datasets'))

app.add_url_rule('/classification/dataset/statistical-analysis',
                 view_func=ClassificationDatasetStatisticalAnalysis.as_view('dataset_statistical_analysis'))

app.add_url_rule('/classification/dataset/box-plot',
                 view_func=ClassificationBoxPlot.as_view('classification_box_plot_data'))

app.add_url_rule('/classification/dataset/histogram',
                 view_func=ClassificationHistogram.as_view('classification_histogram'))

app.add_url_rule('/classification/dataset/count-plot',
                 view_func=ClassificationCountPlot.as_view('classification_count_plot_data'))

app.add_url_rule('/classification/dataset/predict',
                 view_func=DatasetClassificationGetPredict.as_view('classification_dataset_predict'))

app.add_url_rule('/regression/datasets',
                 view_func=RegressionDataset.as_view('regression_datasets'))

app.add_url_rule('/regression/dataset/statistical-analysis',
                 view_func=RegressionDatasetStatisticalAnalysis.as_view('regression_dataset_statistical_analysis'))

app.add_url_rule('/regression/dataset/histogram',
                 view_func=RegressionHistogram.as_view('regression_histogram'))

app.add_url_rule('/regression/dataset/box-plot',
                 view_func=RegressionBoxPlot.as_view('regression_box_plot_data'))

app.add_url_rule('/regression/dataset/count-plot',
                 view_func=RegressionCountPlot.as_view('regression_count_plot_data'))

app.add_url_rule('/regression/dataset/sets',
                 view_func=RegressionDatasetSets.as_view('regression_dataset_sets'))

app.add_url_rule('/regression/train',
                 view_func=RegressionTrain.as_view('regression_train'))

app.add_url_rule('/regression/test',
                 view_func=DataRegressionTest.as_view('regression_test'))

app.add_url_rule('/regression/dataset/predict-fields',
                 view_func=DatasetRegressionGetPredictFields.as_view('regression_dataset_predict_fields'))

app.add_url_rule('/regression/dataset/predict',
                 view_func=DatasetRegressionGetPredict.as_view('regression_dataset_predict'))
