from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_cors import CORS
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch

from generateClassificationData import generate_separable_data, generate_non_separable_data, get_line_params
from classification import LogisticRegression, make_predictions

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


class GeneratedDataClassificationTrain(MethodView):
    def post(self):
        req_data = request.get_json()

        data = (req_data.get('data'))
        max_iter = int(req_data.get('max_iter'))
        learning_rate = float(req_data.get('learning_rate'))
        optimizer_name = req_data.get('optimizer')
        criterion_name = req_data.get('criterion')

        if optimizer_name == '' or optimizer_name == None:
            return jsonify({'error': 'Optimizer name cannot be empty.'}), 400

        if criterion_name == '' or criterion_name == None:
            return jsonify({'error': 'Criterion name cannot be empty.'}), 400

        model = LogisticRegression(2, 1)

        optimizer = getattr(torch.optim, optimizer_name)(
            model.parameters(), lr=learning_rate)
        criterion = getattr(torch.nn, criterion_name)()

        x = np.array([(d['x1'], d['x2']) for d in data])
        y = np.array([d['color'] for d in data])

        result = get_line_params(max_iter, model, optimizer, criterion, x, y)

        return jsonify(result)


app.add_url_rule('/classification/generated/train',
                 view_func=GeneratedDataClassificationTrain.as_view('separable_data_train'))


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
