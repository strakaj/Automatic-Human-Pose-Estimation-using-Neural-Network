import dataset_utils.utils as utils
import json
from scripts.data_loader import MPIIDataset, ToTensor, Rescale
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import torch.optim as optim

config = {}
with open("../dataset_utils/paths.json", 'r') as f:
    paths = json.load(f)

with open("../new_train_set_imgidx.json", 'r') as f:
    train_set_imgidx = json.load(f)

with open("../new_validation_set_imgidx.json", 'r') as f:
    test_set_imgidx = json.load(f)

dataset = utils.Dataset(paths, 1)

trn = transforms.Compose([Rescale((220, 220)), ToTensor()])
batch_size = 1

train_set = MPIIDataset(train_set_imgidx, dataset, transform=trn)
test_set = MPIIDataset(test_set_imgidx, dataset, transform=trn)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 48, 3)
        self.conv5 = nn.Conv2d(48, 48, 3)

        self.fc1 = nn.Linear(48 * 11 * 11, 2048)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(128, 48)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop2(x)

        x = self.fc4(x)
        return x


net = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


def run():
    print(len(train_set))
    print(len(test_set))

    print(train_set[0][0].size(), train_set[0][1].size())

    start_time = time.time()
    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    end_time = time.time()
    print('Finished Training')
    print('Training time: {}'.format(end_time - start_time))

    PATH = './MPII_net.pth'
    torch.save(net.state_dict(), PATH)


def test():
    PATH = './MPII_net.pth'

    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    correct = [0, 0, 0, 0]
    total = 0
    dif = 0.0001
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            total += labels.size(2)
            correct[0] += ((torch.abs(outputs - labels[0]) < dif) * 1).sum().item()
            correct[1] += ((torch.abs(outputs - labels[0]) < dif * 10) * 1).sum().item()
            correct[2] += ((torch.abs(outputs - labels[0]) < dif * 100) * 1).sum().item()
            correct[3] += ((torch.abs(outputs - labels[0]) < dif * 1000) * 1).sum().item()

    print('Accuracy of the network on the test images: {:.2f} % - dif: {}'.format(
        100 * correct[0] / total, dif))
    print('Accuracy of the network on the test images: {:.2f} % - dif: {}'.format(
        100 * correct[1] / total, dif * 10))
    print('Accuracy of the network on the test images: {:.2f} % - dif: {}'.format(
        100 * correct[2] / total, dif * 100))
    print('Accuracy of the network on the test images: {:.2f} % - dif: {}'.format(
        100 * correct[3] / total, dif * 1000))

    correct = [0, 0, 0, 0]
    total = 0
    dif = 0.0001
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            total += labels.size(2)
            correct[0] += ((torch.abs(outputs - labels[0]) < dif) * 1).sum().item()
            correct[1] += ((torch.abs(outputs - labels[0]) < dif * 10) * 1).sum().item()
            correct[2] += ((torch.abs(outputs - labels[0]) < dif * 100) * 1).sum().item()
            correct[3] += ((torch.abs(outputs - labels[0]) < dif * 1000) * 1).sum().item()

    print('Accuracy of the network on the train images: {:.2f} % - dif: {}'.format(
        100 * correct[0] / total, dif))
    print('Accuracy of the network on the train images: {:.2f} % - dif: {}'.format(
        100 * correct[1] / total, dif * 10))
    print('Accuracy of the network on the train images: {:.2f} % - dif: {}'.format(
        100 * correct[2] / total, dif * 100))
    print('Accuracy of the network on the train images: {:.2f} % - dif: {}'.format(
        100 * correct[3] / total, dif * 1000))


if __name__ == '__main__':
    run()
    test()
