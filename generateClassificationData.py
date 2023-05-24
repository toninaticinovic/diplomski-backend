from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
import torch

from classification import train_model


def generate_separable_data(n_samples, centers, train_size):
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers, random_state=100)

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

    return result


def generate_non_separable_data(n_samples, centers, train_size):
    X, y = make_classification(
        n_samples=n_samples, n_features=centers, n_clusters_per_class=2, n_redundant=0)

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

    return result


def get_params(max_iter, model, optimizer, criterion, x, y, dimension):

    line_params = []
    loss_params = []
    for i in range(max_iter):
        # include loss later for testing purposes
        loss = train_model(i, model, torch.Tensor(
            x), torch.Tensor(y), optimizer, criterion)
        if (dimension == 2):
            w, b = model.parameters()
            w1, w2 = w.data[0][0].item(), w.data[0][1].item()
            line_params.append(
                {'w1': w1, 'w2': w2, 'b': b.data[0].item()})
        else:
            loss_params.append({'loss': loss.item(), 'epoch': i})
    if (dimension == 2):
        result = {'line_params': line_params}
    else:
        result = {'loss_params': loss_params}

    return result
