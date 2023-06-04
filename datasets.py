
from flask import jsonify
from flask.views import MethodView

from classification import classification_datasets
from regression import regression_datasets


class ClassificationDataset(MethodView):
    def get(self):
        return jsonify(classification_datasets)


class RegressionDataset(MethodView):
    def get(self):
        return jsonify(regression_datasets)
