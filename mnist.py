import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
batch_size = 64
epochs = 40


def L3_regularizer(weights, lamb, activation, shift):
    l3_reg = Variable(torch.FloatTensor(1), requires_grad=True)
    for W in weights:
        # l3_reg = l3_reg + (W.abs()).sin().clamp(min=0).norm(2)
        l3_reg = l3_reg + lamb * (-(W.abs() - shift).pow(2) + activation).clamp(min=0).norm(2)
        # l3_reg = l3_reg + (W.abs()[W.abs() < 1]).sin().norm()
    return l3_reg

def count_zeros(parameters):
    count = 0
    for W in parameters:
        count += len(W[W < 0.001])
    return count


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01)

data_train = datasets.MNIST('../data', train=True, download=True,
   transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ]))

data_test = datasets.MNIST('../data', train=True, download=True,
   transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ]))

data_train.train_data =  data_train.train_data[:int(len(data_train)/10)]

train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=batch_size, shuffle=True)

def train(epoch, lamb, activation, shift, cuda=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data.cuda()
            target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        regularization = L3_regularizer(model.parameters(), lamb, activation, shift)
        loss = F.nll_loss(output, target) + regularization
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        zeros = count_zeros(model.parameters())
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

def test(cuda=False):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data.cuda()
            target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    zeros = count_zeros(model.parameters())
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Zeros {}'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), zeros))
    return (correct/len(test_loader.dataset), zeros)


def generate_title(lamb, activation, shift, top_accuracy):
    return 'lamb_{lamb}.act_{activation}.shift_{shift}.top_accuracy_{top_accuracy}'.format(
        lamb=lamb, activation=activation, shift=shift, top_accuracy=top_accuracy)

lamb_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
activation_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
shift_values = [0.01, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.20, 0.25]

for lamb in lamb_values:
    for activation in activation_values:
        for shift in shift_values:
            test_accuracies = []
            zeroes_list = []
            print('Running experiment with lamb {} activation {} shift {}'.format(lamb, activation, shift))
            for epoch in range(1, epochs + 1):
                train(epoch, lamb=lamb, activation=activation, shift=shift)
                test_accuracy, zeros = test()
                test_accuracies.append(test_accuracy)
                zeroes_list.append(zeros)
            top_accuracy = test_accuracies[-1]
            title = generate_title(lamb, activation, shift, top_accuracy)
            np.save('results/test_accuracy_' + title, np.array(test_accuracies))
            np.save('results/zeroes_' + title, np.array(zeroes_list))


