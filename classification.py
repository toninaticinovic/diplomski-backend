import torch


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
