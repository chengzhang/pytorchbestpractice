import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


training_data = datasets.FashionMNIST(
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
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


def train(model, data_loader, loss_fn, opt):
    model.train()
    for b, (X, y) in enumerate(data_loader):
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if b % 100 == 0:
            print(f'Train loss: {loss:>7f}, step: {(b+1)*batch_size:>7d}/{len(data_loader.dataset):>7d}')


def evaluate(model, data_loader, loss_fn):
    model.eval()
    loss, correct = .0, 0
    for X, y in data_loader:
        logits = model(X)
        loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    print(f'Eval loss: {loss/len(data_loader):>7f}, acc: {correct/len(data_loader.dataset):>7f}')


n_epoch = 5
for e in range(n_epoch):
    print(f'Epoch {e+1} \n------------------------------------------')
    train(model, train_dataloader, loss_fn, optimizer)
    evaluate(model, test_dataloader, loss_fn)


torch.save(model.state_dict(), 'model.pth')
model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
X, y = test_data[0]
logits = model(X)
pred = logits[0].argmax(0).item()
print(f'Test pred: {pred}, golden: {y}')
