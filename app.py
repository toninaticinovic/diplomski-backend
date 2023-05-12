from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_cors import CORS
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch

app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80)
    app.run(debug=True)


class HelloWorld(MethodView):
    def get(self):
        return jsonify(welcome="hello")


app.add_url_rule("/test", view_func=HelloWorld.as_view("hello_world"))


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def train_model(i, model, x_data, y_data, optimizer, criterion):
    model.train()

    optimizer.zero_grad()

    outputs = model(x_data)

    loss = criterion(torch.squeeze(outputs), y_data)
    loss.backward()

    optimizer.step()

    # if (i+1) % 10 == 0:
    #     print('epoch:', i+1, ',loss=', loss.item())
    return loss


def animate(i, model, x_data, y_data, optimizer, criterion):
    w, b = model.parameters()
    w1 = w.data[0][0]
    w2 = w.data[0][1]
    u = np.linspace(x_data[:, 0].min(), x_data[:, 0].max(), 2)

    loss = train_model(i, model, x_data, y_data, optimizer, criterion)
    y1 = (0.5-b.data-w1*u)/w2

    return {'x': u.tolist(), 'y': y1.tolist(), 'loss': loss}


class SeparableDataClassification(MethodView):
    def post(self):
        req_data = request.get_json()
        n_samples = int(req_data.get('n_samples', '1000'))
        centers = int(req_data.get('centers', '2'))
        train_size = float(req_data.get('train_size', '0.5'))

        X, y = make_blobs(n_samples=n_samples, centers=centers)

        data = [{'x1': x[0], 'x2': x[1],
                 'color': int(c)} for x, c in zip(X, y)]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=1)

        train_data = [{'x1': x[0], 'x2': x[1],
                       'color': int(c)} for x, c in zip(x_train, y_train)]
        test_data = [{'x1': x[0], 'x2': x[1],
                      'color': int(c)} for x, c in zip(x_test, y_test)]

        result = {'data': data, 'train_data': train_data,
                  'test_data': test_data}

        return jsonify(result)


app.add_url_rule('/classification/separable_data',
                 view_func=SeparableDataClassification.as_view('separable_data'))


class SeparableDataClassificationTrain(MethodView):
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

        line_params = []
        for i in range(max_iter):
            # include loss later for testing purposes
            loss = train_model(i, model, torch.Tensor(
                x), torch.Tensor(y), optimizer, criterion)
            w, b = model.parameters()
            w1, w2 = w.data[0][0].item(), w.data[0][1].item()
            line_params.append(
                {'w1': w1, 'w2': w2, 'b': b.data[0].item()})

        result = {'line_params': line_params}
        return jsonify(result)


app.add_url_rule('/classification/separable_data/train',
                 view_func=SeparableDataClassificationTrain.as_view('separable_data_train'))
