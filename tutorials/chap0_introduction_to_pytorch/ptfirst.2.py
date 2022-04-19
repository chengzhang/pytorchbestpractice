import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64
train_dl = DataLoader(train_data, batch_size=batch_size)
test_dl = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(model, dl, loss_fn, optimizer):
    for b, (X, y) in enumerate(dl):
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b % 100 == 0:
            print(f'Train Loss {loss.item():>7f}, Step {(b+1)*batch_size:>5d}/{len(dl.dataset):>5d}')


def evaluate(model, dl, loss_fn):
    loss, correct = 0.0, 0
    for X, y in dl:
        logits = model(X)
        loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    print(f'Eavl \n\tLoss {loss/len(dl):>7f}, Acc {correct/len(dl.dataset):>7f}')


n_epoch = 1
for e in range(n_epoch):
    print(f'Epoch {e+1} \n------------------------------------------')
    train(model, train_dl, loss_fn, optimizer)
    evaluate(model, test_dl, loss_fn)


torch.save(model.state_dict(), 'model.pth')
model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
pdb.set_trace()
X, y = test_data[0]
logits = model(X)
pred = logits[0].argmax(0)
print(f'pred {pred}, golden {y}')
