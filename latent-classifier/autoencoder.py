import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt


EMBEDDING_DIM = 50
BATCH_SIZE = 128
HIDDEN_LAYER = 256

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, HIDDEN_LAYER)
        self.fc2 = nn.Linear(HIDDEN_LAYER, EMBEDDING_DIM)
        # might add another layer

    def forward(self,x):
        h1 = F.relu(self.fc1(x)) # might want to remove
        return F.relu(self.fc2(h1))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(EMBEDDING_DIM, HIDDEN_LAYER)
        self.fc2 = nn.Linear(HIDDEN_LAYER, 28 * 28)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class FeatureClassifier(nn.Module):
    def __init__(self, encoder):
        super(FeatureClassifier, self).__init__()
        freeze_model_params(encoder)
        self.encoder = encoder
        self.fc1= nn.Linear(EMBEDDING_DIM, 10)

    def forward(self, x):
        h1 = self.encoder(x)
        return F.log_softmax(self.fc1(h1), dim=1)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, HIDDEN_LAYER)
        self.fc2 = nn.Linear(HIDDEN_LAYER, EMBEDDING_DIM)
        self.fc3 = nn.Linear(EMBEDDING_DIM, 10)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return F.log_softmax(self.fc3(h2))


encoder = Encoder()
decoder = Decoder()

ae = Autoencoder(encoder, decoder)


train_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

test_set = datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

train_set.train_data = train_set.train_data[:1000]

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

view_data = test_set.test_data[:16].view(-1, 28 * 28).numpy()
fig = plt.figure(figsize=(4,4))
classifier = Classifier()

for i in range(1, 17):
    fig.add_subplot(4,4,i)
    plt.imshow(view_data[i-1].reshape(28,28))
plt.title('Original')

def train_ae(model):
    print('Training AE')
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.view(-1, 28*28)), Variable(target)
        output_images = model(data)
        optimizer.zero_grad()
        loss = loss_func(output_images, data)
        loss.backward()
        optimizer.step()

def render_test(model):
    data = Variable(torch.Tensor(view_data))
    output_images = model(data)
    fig = plt.figure(figsize=(4, 4))
    for i in range(1, 17):
        fig.add_subplot(4, 4, i)
        image = output_images.data[i-1].numpy()
        plt.imshow(image.reshape(28, 28))
    plt.title('Generated')
    plt.show()


def train(model, epoch):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_func = nn.NLLLoss()
    for idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.view(-1, 28*28)), Variable(target)
        logp = model(data)
        optimizer.zero_grad()
        loss = loss_func(logp, target)
        loss.backward()
        optimizer.step()
    print(epoch, loss.data)

def test(model):
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data.view(-1, 28 * 28), volatile=True), Variable(target)
        logp = model(data)
        pred = logp.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print('Test Accuracy', correct / len(test_loader.dataset))
    return correct / len(test_loader.dataset)

def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

def count_params(model):
    count = 0
    for param in model.parameters():
        nn = 1
        for s in list(param.size()):
            nn *= s
        count += nn
    return count

EPOCHS = 25

# 0.9449 accuracy

# classifier_accuracies = []
# for epoch in range(EPOCHS):
#     train(classifier, epoch)
#     accuracy = test(classifier)
#     classifier_accuracies.append(accuracy)

# np.save('classifier_accuracies', np.array(classifier_accuracies))


# for epoch in range(EPOCHS):
#     train_ae(ae)

# render_test(ae)


# torch.save(encoder.state_dict(), 'encoder.pt')
#
encoder = Encoder()
encoder.load_state_dict(torch.load('encoder.pt'))
feature_classifier = FeatureClassifier(encoder)

features_accuracies = []
for epoch in range(EPOCHS):
    train(feature_classifier, epoch)
    accuracy = test(feature_classifier)
    features_accuracies.append(accuracy)

np.save('features_accuracies', np.array(features_accuracies))



