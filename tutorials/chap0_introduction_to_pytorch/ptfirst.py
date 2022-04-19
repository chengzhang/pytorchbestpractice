import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
)


batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(model, dataloader, loss_fn, opt):
    model.train()
    size = len(dataloader)
    for b, (X, y) in enumerate(dataloader):
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if b % 100 == 0:
            print(f'Train Loss {loss.item():>7f}, Step {b*len(X):>5d}/{size*len(X):>5d}')


def eval(model, dataloader, loss_fn):
    model.eval()
    size = len(dataloader)
    loss, correct = 0.0, 0
    for X, y in dataloader:
        logits = model(X)
        loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    loss /= size
    correct /= len(dataloader.dataset)
    print(f'Test loss {loss:>7f}, Acc: {correct:>7f}')


n_epoch = 1
for i in range(n_epoch):
    print(f'Epoch {i+1} \n------------------------------------------')
    train(model, train_dataloader, loss_fn, opt)
    eval(model, test_dataloader, loss_fn)


torch.save(model.state_dict(), 'model.pth')
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
X, y = test_data[0]
with torch.no_grad():
    logits = loaded_model(X)
    pred = logits[0].argmax(0)
    print(f'pred: {pred}, golden: {y}')
