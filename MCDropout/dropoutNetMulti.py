import torch
from torchvision import datasets, transforms
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

import numpy as np
import sys
import shap

batch_size = 128
num_epochs = 20
device = torch.device('cuda')
inference_mode = 'all'
train_mode = 'test'
forward_passes = 50
path = r"..\models\multi" + ".pth"


class Net(nn.Module):
    def __init__(self, forward_passes: int = 20, mode: str = 'point') -> None:
        """Initialize the network."""
        super(Net, self).__init__()
        
        self.forward_passes = forward_passes
        self.mode = mode

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.3),        
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.3),        
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.mode == 'point':
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)

        elif self.mode == 'total_entropy':
            x = self.conv_layers
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            x = self.total_entropy(x)

        elif self.mode == 'ep_entropy':
            ep_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'ep_entropy')
            x = ep_entropy

        elif self.mode == 'al_entropy':
            conv_out = self.conv_layers(x)
            conv_out = conv_out.view(-1, 320)
            fc_out = self.fc_layers(conv_out)
            total_entropy = self.total_entropy(fc_out)

            ep_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'ep_entropy')
            al_entropy = total_entropy - ep_entropy
            x = al_entropy
            # print(ep_entropy)
            # print(al_entropy)
            # print(total_entropy)

        else:
            x = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, self.mode)

        return x
    
    def total_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate total entropy."""
        entropy = -torch.sum(x * torch.log(x), dim=-1)
        return entropy[:, None]

    def enable_dropout(self) -> None:
        """Enable dropout layers during test-time."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_monte_carlo_predictions(self, x: torch.Tensor, batch: int, forward_passes: int, mode: str) -> torch.Tensor:
        """Get Monte Carlo predictions."""
        n_classes = 10
        n_samples = batch
        device = x.device
        dropout_predictions = torch.empty((0, n_samples, n_classes), device=device)
        
        for i in range(forward_passes):
            predictions = torch.empty((0, n_classes), device=device)
            self.enable_dropout()
            conv_out = self.conv_layers(x)
            reshape_out = conv_out.view(-1, 320)
            predictions = self.fc_layers(reshape_out)
            dropout_predictions = torch.cat((dropout_predictions, predictions.unsqueeze(0)), dim=0)

        # Calculate mean across multiple MCD forward passes
        mean = dropout_predictions.mean(dim=0)

        # Calculate entropy across multiple MCD forward passes
        entropy = -torch.sum(dropout_predictions * torch.log(dropout_predictions), dim=-1)
        entropy = entropy.mean(dim=0)
        
        if mode == 'mean':
            return mean
        if mode == 'ep_entropy':
            return entropy[:, None]
        

def train(model: torch.nn.Module, 
          device: torch.device, 
          train_loader: DataLoader, 
          optimizer: Optimizer, 
          epoch: int) -> None:
    """
    Trains the model.

    Parameters:
    - model: The neural network model to be trained.
    - device: The device (CPU/GPU) where the model will be trained.
    - train_loader: DataLoader object containing the training data.
    - optimizer: Optimizer object.
    - epoch: Current epoch number.

    Returns:
    - None
    """
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def create_shap_plots(model: torch.nn.Module, 
                      background: Tensor, 
                      test_images: Tensor, 
                      mode: str) -> None:
    """
    Create SHAP plots for the model predictions.

    Parameters:
    - model: The trained neural network model.
    - background: Background data for the SHAP explainer.
    - test_images: Images to be explained.
    - mode: The mode for SHAP plotting.

    Returns:
    - None
    """
    explainer = shap.DeepExplainer(model, background)
    shap_values = np.asarray(explainer.shap_values(test_images))

    test_images_cpu = test_images.to(torch.device('cpu'))

    if mode in ['al_entropy', 'total_entropy', 'ep_entropy']:
        shap_entropy_numpy = np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2)
        test_entropy_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_entropy_numpy, -test_entropy_numpy)
    elif mode in ['mean', 'point']:
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images_cpu.numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)

def create_dataset(x: int = 1, y: int = 7) -> Tuple[DataLoader, DataLoader, str]:
    """
    Create a dataset for training and testing.

    Parameters:
    - x: Label 1 for the dataset.
    - y: Label 2 for the dataset.

    Returns:
    - Tuple containing train_loader, test_loader, and model path.
    """
    train_dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    train_idx1 = torch.tensor(train_dataset.targets) == x
    train_idx2 = torch.tensor(train_dataset.targets) == y

    train_indices_1 = train_idx1.nonzero().reshape(-1)
    train_indices_2 = train_idx2.nonzero().reshape(-1)

    for i in train_indices_1:
        train_dataset.targets[i] = 0
    for j in train_indices_2:
        train_dataset.targets[j] = 1

    train_mask = train_idx1 | train_idx2
    train_indices = train_mask.nonzero().reshape(-1)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    test_idx1 = torch.tensor(test_dataset.targets) == x
    test_idx2 = torch.tensor(test_dataset.targets) == y

    test_indices_1 = test_idx1.nonzero().reshape(-1)
    test_indices_2 = test_idx2.nonzero().reshape(-1)

    for i in test_indices_1:
        test_dataset.targets[i] = 0
    for j in test_indices_2:
        test_dataset.targets[j] = 1

    test_mask = test_idx1 | test_idx2
    test_indices = test_mask.nonzero().reshape(-1)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)
    path = r"..\models\MCDropout" + str(x) + str(y) + ".pth"
    return train_loader, test_loader, path


if __name__ == '__main__':
    path = r"..\models\multi" + ".pth"

    train_dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    test_dataset = datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=True)
#################################################################################

    trainPointModel = Net(mode="point").to(device)
    optimizer = optim.SGD(trainPointModel.parameters(), lr=0.01, momentum=0.5)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:1000].to(device)
    test_images = images[0:5]    

    if train_mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(trainPointModel, device, train_loader, optimizer, epoch)
        torch.save(trainPointModel.state_dict(), path)

    if train_mode == 'test':
        if inference_mode == 'point':
            #Initialise model(s)
            pointModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            pointModel.load_state_dict(torch.load(path))
            out = pointModel(test_images.to(device))

            create_shap_plots(pointModel, background, test_images, inference_mode)

        elif inference_mode == 'mean':
            #Initialise model(s)
            meanModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            meanModel.load_state_dict(torch.load(path))
            out = meanModel(test_images.to(device))

            create_shap_plots(meanModel, background, test_images, inference_mode)

        elif inference_mode == 'al_entropy':
            #Initialise model(s)
            alEntropyModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            alEntropyModel.load_state_dict(torch.load(path))

            entropies = alEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(alEntropyModel, background, test_images, inference_mode)

        elif inference_mode == 'total_entropy':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            totalEntropyModel.load_state_dict(torch.load(path))

            entropies = totalEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(totalEntropyModel, background, test_images, inference_mode)

        elif inference_mode == 'all':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=forward_passes, mode='total_entropy').to(device)
            totalEntropyModel.load_state_dict(torch.load(path))
            alEntropyModel = Net(forward_passes=forward_passes, mode='al_entropy').to(device)
            alEntropyModel.load_state_dict(torch.load(path))
            epEntropyModel = Net(forward_passes=forward_passes, mode='ep_entropy').to(device)
            epEntropyModel.load_state_dict(torch.load(path))
            pointModel = Net(forward_passes=forward_passes, mode='point').to(device)
            pointModel.load_state_dict(torch.load(path))

            #Filter subsample for high entropy
            entropies = totalEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(totalEntropyModel, background, test_images, 'total_entropy')
            create_shap_plots(alEntropyModel, background, test_images, 'al_entropy')
            create_shap_plots(epEntropyModel, background, test_images, 'ep_entropy')
            create_shap_plots(pointModel, background, test_images, 'point')

        else:
            print("Invalid inference mode selected!")
            