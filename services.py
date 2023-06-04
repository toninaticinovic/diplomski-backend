
import torch


def train_model(i, model, x_data, y_data, optimizer, criterion):
    model.train()

    optimizer.zero_grad()

    outputs = model(x_data)

    if isinstance(criterion, torch.nn.MSELoss):
        loss = torch.sqrt(criterion(torch.squeeze(outputs), y_data))
    else:
        loss = criterion(torch.squeeze(outputs), y_data)

    loss.backward()

    optimizer.step()

    # if (i+1) % 10 == 0:
    #   print('epoch:', i+1, ',loss=', loss.item())
    return loss


def get_params(max_iter, model, optimizer, criterion, x, y, dimension):
    line_params = []
    loss_params = []
    for i in range(max_iter):
        loss = train_model(i, model, torch.Tensor(
            x), torch.Tensor(y), optimizer, criterion)
        if (dimension == 2):
            w, b = model.parameters()
            w1, w2 = w.data[0][0].item(), w.data[0][1].item()
            line_params.append(
                {'w1': w1, 'w2': w2, 'b': b.data[0].item()})
        else:
            loss_params.append({'loss': loss.item(), 'epoch': i})
    w, b = model.parameters()
    latest_params = {'w': w.detach().numpy().tolist(),
                     'b': b.detach().numpy().tolist()}

    if (dimension == 2):
        result = {'line_params': line_params}
    else:
        result = {'loss_params': loss_params}

    return {"result": result, "latest_params": latest_params}
