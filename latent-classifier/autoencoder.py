import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


EMBEDDING_DIM = 32
BATCH_SIZE = 32

class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(784, embedding_dim)
        # might add another layer

    def forward(self,x):
        return F.relu(self.linear(x)) # might want to remove


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(embedding_dim, 784)

    def forward(self, x):
        return torch.sigmoid(self.linear())


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


encoder = Encoder(EMBEDDING_DIM)
decoder = Decoder(EMBEDDING_DIM)

model = Autoencoder(encoder, decoder)
optimizer = optim.Adam(model.parameters(), lr=0.01)


train_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

def train():
    loss_func = nn.MSELoss()

    for idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output_images = model(data)
        optimizer.zero_grad()
        loss = loss_func(output_images, data)
        loss.backwards()
        if idx % 100:
            print(idx, loss)


train()