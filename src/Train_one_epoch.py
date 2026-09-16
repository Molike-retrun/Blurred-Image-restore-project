import torch
from torch import nn

def train_one_epoch(
    model,
    train_loader,
    device,
    criterion,
    optimizer
):

    model.train()
    total_loss = 0.0

    for blurred, clean in train_loader:

        blurred = blurred.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        restored = model(blurred)
        loss = criterion(restored, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss/len(train_loader)
    

    return train_loss