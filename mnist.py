import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# class Zelu(nn.Module):
#     def __init__(self, lower=0.1, upper=0.1):
#         self._lower = lower
#         self._upper = upper
#
#
#     def forward(self, input):
#         right = F.threshold(input - self._upper, self._upper, 0)
#         left = -F.threshold(self._lower - input , self._lower, 0)
#         return right + left

ZELU_THRESHOLD = 3

def zelu(input, lower=-ZELU_THRESHOLD, upper=ZELU_THRESHOLD):
    right = F.threshold(input - upper, upper, 0)
    left = -F.threshold(lower - input, lower, 0)
    return right + left


class Net(nn.Module):
    def __init__(self, use_zelu=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self._use_zelu = use_zelu

    def forward(self, x):
        if not self._use_zelu:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        else:
            x = zelu(F.max_pool2d(self.conv1(x), 2))
            x = zelu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class DNNet(nn.Module):
    def __init__(self, use_zelu=False):
        super(DNNet, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self._use_zelu = use_zelu

    def forward(self, x):
        x = x.view(-1, 784)
        if self._use_zelu:
            x = zelu(self.fc1(x))
            # print('layer 1,', x.max(), x.min())
            x = zelu(self.fc2(x))
            # print('layer 2,', x.max(), x.min())
        else:
            x = F.relu(self.fc1(x))
            # print('layer 1,', x.max(), x.min())
            x = F.relu(self.fc2(x))
            # print('layer 2,', x.max(), x.min())

        return F.log_softmax(x)

batch_size = 64
epochs = 20
training_set_ratio = 10

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

model = DNNet(use_zelu=ZELU_THRESHOLD != 0)

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

data_train.train_data =  data_train.train_data[:int(len(data_train)/training_set_ratio)]

train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=batch_size, shuffle=True)



def train(epoch, cuda=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # regularization = L3_regularizer(model.parameters(), lamb, activation, shift)
        # loss = F.nll_loss(output, target) + regularization
        loss = F.nll_loss(output, target)
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
            data, target = data.cuda(), target.cuda()
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

for epoch in range(1, epochs + 1):
    train(epoch)
    test()
    # test_accuracies.append(test_accuracy)
    # zeroes_list.append(zeros)

# for lamb in lamb_values:
#     for activation in activation_values:
#         for shift in shift_values:
#             test_accuracies = []
#             zeroes_list = []
#             print('Running experiment with lamb {} activation {} shift {}'.format(lamb, activation, shift))
#             for epoch in range(1, epochs + 1):
#                 train(epoch, lamb=lamb, activation=activation, shift=shift)
#                 test_accuracy, zeros = test()
#                 test_accuracies.append(test_accuracy)
#                 zeroes_list.append(zeros)
#             top_accuracy = test_accuracies[-1]
#             title = generate_title(lamb, activation, shift, top_accuracy)
#             np.save('results/test_accuracy_' + title, np.array(test_accuracies))
#             np.save('results/zeroes_' + title, np.array(zeroes_list))




