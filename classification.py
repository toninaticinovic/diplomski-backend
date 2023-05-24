import torch


datasets = [
    {'label': 'Autentifikacija novčanica', 'value': 'bank_note_authentication',
     'description': 'Skup podataka iz kojeg se izgrađuje model klasifikacije za predviđanje autentičnosti novčanica na temelju zadanih značajki (variance, skewness, curtosis, entropy).'},
    {'label': 'Pima Indians Dijabetes', 'value': 'diabetes',
     'description': 'Skup podataka iz kojeg se izgrađuje model klasifikacije za dijagnosticiranje je li pacijent dijabetičar ili ne, na temelju određenih dijagnostičkih mjera (npr. broj trudnoća, razina glukoze u krvi, ...). Sve pacijentice su žene iz indijskog plemena Pima, koje imaju najmanje 21 godinu.'},
    {'label': 'Habermanov skup podataka o preživljavanju', 'value': 'haberman',
     'description': 'Skup podataka sadrži slučajeve iz studije koja je provedena između 1958. i 1970. u bolnici Billings Sveučilišta u Chicagu preživljavanje pacijenata koji su bili podvrgnuti operaciji raka dojke. Značajke koje se uzimaju u obzir su starost pacijenta, godina operacije i broj otkrivenih pozitivnih aksilarnih čvorova.'},
    {'label': 'Pitkost vode', 'value': 'water_potability', 'description': 'Skup podataka koji sadrži informacije o različitim parametrima kvalitete vode kako bi se utvrdilo je li voda pitka ili ne. Skup podataka pruža opažanja o različitim kemijskim i fizičkim karakteristikama uzoraka vode iz različitih izvora poput pH vode ili tvrdoće vode.', }
]


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


def make_predictions(model, x_test):
    with torch.no_grad():
        predictions = model(torch.Tensor(x_test)).round()
    return predictions.numpy().tolist()
