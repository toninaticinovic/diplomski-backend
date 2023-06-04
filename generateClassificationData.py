from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split


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
