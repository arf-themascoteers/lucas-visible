import torch
from torch.utils.data import DataLoader
from model_ann import ANN


def train(device, ds, model=None, num_epochs=300):
    torch.manual_seed(0)
    batch_size = 600
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    x_size = ds.get_x().shape[1]
    if model is None:
        model = ANN(size = x_size)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    n_batches = int(len(ds)/batch_size) + 1
    for epoch in range(num_epochs):
        batch_number = 0
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            x1, x2 = model(x)
            loss = model.calculate_loss(x1, x2, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_number += 1
            #print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    torch.save(model, "ann.h5")
    return model
