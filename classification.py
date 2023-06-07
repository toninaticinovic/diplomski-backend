from flask import jsonify, request
from flask.views import MethodView
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch

from services import get_params

classification_datasets = [
    {'label': 'Autentifikacija novčanica', 'value': 'bank_note_authentication',
     'description': 'Skup podataka iz kojeg se izgrađuje model klasifikacije za predviđanje autentičnosti novčanica na temelju zadanih značajki (variance, skewness, curtosis, entropy).'},
    {'label': 'Pima Indians Dijabetes', 'value': 'diabetes',
     'description': 'Skup podataka iz kojeg se izgrađuje model klasifikacije za dijagnosticiranje je li pacijent dijabetičar ili ne, na temelju određenih dijagnostičkih mjera (npr. broj trudnoća, razina glukoze u krvi, ...). Sve pacijentice su žene iz indijskog plemena Pima, koje imaju najmanje 21 godinu.'},
    {'label': 'Habermanov skup podataka o preživljavanju', 'value': 'haberman',
     'description': 'Skup podataka sadrži slučajeve iz studije koja je provedena između 1958. i 1970. u bolnici Billings Sveučilišta u Chicagu preživljavanje pacijenata koji su bili podvrgnuti operaciji raka dojke. Značajke koje se uzimaju u obzir su starost pacijenta, godina operacije i broj otkrivenih pozitivnih aksilarnih čvorova.'},
]

classification_dataset_predictions = {
    'bank_note_authentication': {
        0: 'Novčanica je autentična',
        1: 'Novčanica nije autentična'
    },
    'diabetes': {
        0: 'Pacijent nije dijabetičar',
        1: 'Pacijent je dijabetičar'
    },
    'haberman': {
        0: 'Pacijent je preživio 5 godina ili duže nakon operacije',
        1: 'Pacijent nije preživio 5 godina nakon operacije'
    }
}


def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    return data[(data < (Q1 - 1.5 * IQR)) | (data > Q3 + 1.5 * IQR)]


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def make_predictions(model, x_test):
    with torch.no_grad():
        predictions = model(torch.Tensor(x_test)).round()
    return predictions.numpy().tolist()


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

        if dataset == None:
            dataset = 'generated'

        path = 'models/classification/' + dataset + '.pt'
        torch.save(model.state_dict(), path)

        return jsonify(result)


class DataClassificationTest(MethodView):
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
        model = LogisticRegression(dimension, 1)

        if dataset == None:
            dataset = 'generated'

        path = 'models/classification/' + dataset + '.pt'
        model.load_state_dict(torch.load(path))
        model.eval()

        predictions_train = make_predictions(model, x_train)
        predictions_test = make_predictions(model, x_test)

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
