from flask import jsonify, request
from flask.views import MethodView

from generateClassificationData import generate_non_separable_data, generate_separable_data


class SeparableDataClassification(MethodView):
    def post(self):
        req_data = request.get_json()
        n_samples = int(req_data.get('n_samples', '1000'))
        centers = int(req_data.get('classes', '2'))
        train_size = float(req_data.get('train_size', '0.5'))

        result = generate_separable_data(n_samples, centers, train_size)

        return jsonify(result)


class NonSeparableDataClassification(MethodView):
    def post(self):
        req_data = request.get_json()
        n_samples = int(req_data.get('n_samples', '1000'))
        centers = int(req_data.get('classes', '2'))
        train_size = float(req_data.get('train_size', '0.5'))

        result = generate_non_separable_data(n_samples, centers, train_size)

        return jsonify(result)
