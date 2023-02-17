import torch
from torch.utils.data import DataLoader
from model_ann import ANN
import time


def train(device, ds, model=None, nn_config=None):
    torch.manual_seed(0)
    TEST = False
    if nn_config is None:
        nn_config = {"num_epochs":300, "batch_size":6000,
                     "lr" : 0.001, "mid" : [200]
                     }
    num_epochs = nn_config["num_epochs"] if "num_epochs" in nn_config else 300
    batch_size = nn_config["batch_size"] if "batch_size" in nn_config else 6000
    lr = nn_config["lr"] if "lr" in nn_config else 0.001
    mid = nn_config["mid"] if "mid" in nn_config else [200]
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    x_size = ds.get_x().shape[1]
    if model is None:
        model = ANN(size = x_size, mid=mid)
    if TEST:
        print(num_epochs, batch_size, lr, mid)
        print(model)
        return model
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = torch.nn.MSELoss(reduction='sum')
    n_batches = int(len(ds)/batch_size) + 1
    batch_number = 0
    loss = None
    start = time.time()
    for epoch in range(num_epochs):
        batch_number = 0
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_hat = y_hat.reshape(-1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_number += 1
            #print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    return model

#
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train(device)