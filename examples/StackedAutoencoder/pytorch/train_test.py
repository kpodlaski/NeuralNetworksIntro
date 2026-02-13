'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import os

import torch
import time
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as torchfun

from examples.CNN.pytorch.cnn_model import ConvNet
from examples.DenseNetwork.pytorch.dense_model import DenseNet
from common.pytorch.ml_wrapper import ML_Wrapper
from examples.StackedAutoencoder.pytorch.show_effects import show_effects
from examples.StackedAutoencoder.pytorch.stacked_autoencoder import StackedAutoencoder



print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

base_path = "../../.."
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
latent_dim = 32

model_version = 1
pre_training_epochs = 5
clear_grads_in_between = True

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=base_path+'/datasets/',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=base_path+'/datasets/',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size =batch_size_test, shuffle=False)

model = StackedAutoencoder(latent_dim, stack_times=2, version=model_version).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Starting pretraining")

for epoch in range(pre_training_epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        rot_images = torchfun.rotate(images, angle=45)
        #rot_images = torch.rot90(images, k=1, dims=(2, 3))
        outputs = model.singleAE(images)
        loss = criterion(outputs, rot_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
print(f"Finished pretraining [{pre_training_epochs} epochs]")

print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

show_effects(model, example_data, device, "half_time", model_version, base_path)

if clear_grads_in_between:
    model.zero_grad()

for epoch in range(n_epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        rot_images = torch.rot90(images, k=1, dims=(2, 3))
        outputs = model(images)
        loss = criterion(outputs, rot_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss / len(train_loader):.4f}")

print(f"Finished training [{pre_training_epochs}+{n_epochs} epochs]")
show_effects(model, example_data, device, "final", model_version, base_path)


summary(model, input_size=(1, 28, 28))
