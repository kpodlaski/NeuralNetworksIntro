import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torchsummary import summary


class ML_Wrapper():
    def __init__(self, network, optimizer, base_path, device):
        self.network=network
        self.network.to(device)
        self.optimizer = optimizer
        self.device = device
        self.base_path = base_path
        self.train_losses = []
        self.val_losses = []

    def train(self, epoch, data_loader, val_loader = None):
        self.network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            train_loss+= loss.item()
            loss.backward()
            self.optimizer.step()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, train_loss))
        self.train_losses.append(train_loss)
        if (val_loader):
            self.test(val_loader, val_test=True)

    def test(self, data_loader, val_test=False):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.network(data)
                test_loss += F.nll_loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        if (val_test):
            self.val_losses.append(test_loss)
        else:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)))

    def save_model(self, fname):
        torch.save(self.network.state_dict(), self.base_path + '/out/{}_model.pth'.format(fname))
        #torch.save(optimizer.state_dict(), base_path+'/results/optimizer.pth')

    def training_plot(self):
        fig = plt.figure()
        plt.plot([*range(len(self.train_losses))], self.train_losses, color='blue')
        if (len(self.val_losses))>0:
            plt.plot([*range(len(self.val_losses))], self.val_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('negative log likelihood loss')
        fig.show()

    def summary(self, size):
        x = torch.Tensor(size).to(self.device)
        summary(self.network, size)