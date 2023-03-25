'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from examples.Conv1D.pytorch.cnn1d_model import Conv1DNet
from common.pytorch.ml_wrapper import ML_Wrapper
from examples.Conv1D.read_dataset import read_acc_data_one_hot, read_acc_data

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
base_path = "../../../"
n_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10




network = Conv1DNet()

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device)
#ml.summary((1,128))

train_X, train_Y, test_X, test_Y = read_acc_data()
train_Y = train_Y -1
test_Y = test_Y -1
train_X = train_X.reshape(-1,1,128)
test_X = test_X.reshape(-1,1,128)
print(torch.Tensor(train_Y).long().reshape(-1))
print(torch.Tensor(train_Y).reshape(-1).shape)
train_loader = torch.utils.data.TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y.reshape(-1)).long())
test_loader = torch.utils.data.TensorDataset(torch.Tensor(test_X), torch.Tensor(test_Y.reshape(-1)).long())
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size_train, shuffle=True, **data_loader_kwargs)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
print(example_targets.shape)



print("Start training")
for epoch in range(1, n_epochs + 1):
  ml.train(epoch, train_loader, test_loader)
ml.save_model("dense_network")

print("Test on training set:")
ml.test(train_loader)
print("Test on test set:")
ml.test(test_loader)

ml.training_plot()


print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
  example_data =example_data.to(device)
  network.to(device)
  output = ml.network(example_data)

example_data =example_data.cpu()
