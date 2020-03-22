import torch
from tqdm import tqdm

train_losses = []
train_acc = []

def train(net, device, trainloader, optimizer, criterion, epoch):
  net.train()
  pbar = tqdm(trainloader)
  running_loss = 0.0
  for i, (data, target) in enumerate(pbar):
        # get the inputs
        correct = 0
        processed = 0
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Predict
        y_pred = net(data)
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Epoch= {epoch} Loss={loss.item()} Batch_id={i} Accuracy={100*correct/processed:0.2f}')
        pbar.update(1)
  train_acc = 100*correct/processed
  return train_acc