from flask import jsonify, request
from flask.views import MethodView
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from services import get_params

regression_datasets = [
    {'label': 'Skup podataka o oglašavanju (eng.advertising)', 'value': 'advertising',
     'description': 'Skup podataka model koji sadrži podatke o ulaganjima u oglašavanju na različitim medijskim platformama izraženo u tisućama dolara. Izlazna varijabla je prodaja koja predstavlja količinu prodanih proizvoda izraženih u tisućama jedinica proizvoda.'},
    {'label': 'Skup podataka o iznajmljivanju bicikala', 'value': 'bike_rent',
     'description': 'Skup podataka sadrži broj iznajmljenih bicikala po danu, zajedno s raznim vremenskim i sezonskim informacijama.'},
    {'label': 'Skup podataka o ribama', 'value': 'fish',
     'description': 'Ovaj skup podataka sadrži podatke o 7 uobičajenih različitih vrsta riba u prodaji na ribarnici. Cilj je predvidjeti težinu ribe na temelju mjerenja.'},
]


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def make_predictions(model, x_test):
    with torch.no_grad():
        predictions = model(torch.Tensor(x_test))
    return predictions.numpy().tolist()


class RegressionTrain(MethodView):
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
        model = LinearRegressionModel(dimension, 1)

        optimizer = getattr(torch.optim, optimizer_name)(
            model.parameters(), lr=learning_rate)
        criterion = getattr(torch.nn, criterion_name)()

        result = get_params(max_iter, model, optimizer,
                            criterion, x, y, dimension)

        path = 'models/regression/' + dataset + '.pt'
        torch.save(model.state_dict(), path)

        return jsonify(result)


class DataRegressionTest(MethodView):
    def post(self):
        req_data = request.get_json()

        test_data = req_data.get('test_data')
        train_data = req_data.get('train_data')
        dataset = req_data.get('dataset')

        if train_data is not None and test_data is not None and dataset is None:
            x_train = np.array([(d['x1'], d['x2']) for d in train_data])
            y_train = np.array([d['color'] for d in train_data])
            x_test = np.array([(d['x1'], d['x2']) for d in test_data])
            y_test = np.array([d['color'] for d in test_data])

        elif train_data is not None and test_data is not None and dataset is not None:
            x_train = []
            y_train = []

            for d in train_data:
                x_values = [v for k, v in d.items() if k != 'target']
                x_train.append(x_values)
                y_train.append(d['target'])
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            x_test = []
            y_test = []

            for d in test_data:
                x_values = [v for k, v in d.items() if k != 'target']
                x_test.append(x_values)
                y_test.append(d['target'])
            x_test = np.array(x_test)
            y_test = np.array(y_test)

        dimension = x_test.shape[1]
        model = LinearRegressionModel(dimension, 1)

        path = 'models/regression/' + dataset + '.pt'
        model.load_state_dict(torch.load(path))
        model.eval()

        predictions_train = make_predictions(model, x_train)
        predictions_test = make_predictions(model, x_test)

        r2_score_train = r2_score(y_train, predictions_train)
        r2_score_test = r2_score(y_test, predictions_test)

        mse_train = mean_squared_error(y_train, predictions_train)
        mse_test = mean_squared_error(y_test, predictions_test)

        result = {'mse_test': mse_test,
                  'r2_score_test': r2_score_test,
                  'mse_train': mse_train,
                  'r2_score_train': r2_score_train}
        return jsonify(result)
