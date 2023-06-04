from flask import jsonify, request
from flask.views import MethodView
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def sets(X, y, train_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size)

    train_data = [dict(x, target=c) for x, c in zip(
        X_train.to_dict(orient='records'), y_train)]
    test_data = [dict(x, target=c) for x, c in zip(
        X_test.to_dict(orient='records'), y_test)]

    return train_data, test_data


class ClassificationDatasetSets(MethodView):
    def post(self):
        req_data = request.get_json()
        train_size = float(req_data.get('train_size', '0.5'))
        dataset_name = req_data.get('dataset')

        data = pd.read_csv('datasets/classification/' + dataset_name + '.csv')

        X = data.drop(columns=['class'])
        y = data['class']

        train_data, test_data = sets(X, y, train_size)

        return jsonify({
            'train_data': train_data,
            'test_data': test_data
        })


class RegressionDatasetSets(MethodView):
    def post(self):
        req_data = request.get_json()
        train_size = float(req_data.get('train_size', '0.5'))
        dataset_name = req_data.get('dataset')
        checkbox = bool(req_data.get('checkbox'))

        data = pd.read_csv('datasets/regression/' + dataset_name + '.csv')

        numerical_columns = data.select_dtypes(
            include=['float64', 'int64']).columns
        categorical_columns = data.select_dtypes(include='object').columns

        if checkbox:
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

                # Exclude rows with outliers
                data = data[~data[column].isin(outliers)]

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        label_encoder = LabelEncoder()
        for column in categorical_columns:
            X[column] = label_encoder.fit_transform(X[column])

        train_data, test_data = sets(X, y, train_size)

        return jsonify({
            'train_data': train_data,
            'test_data': test_data
        })
