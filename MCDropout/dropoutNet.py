# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision import datasets, transforms
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from typing import Tuple
import argparse

import numpy as np
import sys
import shap

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
            nn.Linear(50, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.mode == 'point':
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)

        elif self.mode == 'total_entropy':
            total_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'total_entropy')
            x = total_entropy

        elif self.mode == 'al_entropy':
            al_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'al_entropy')
            x = al_entropy

        elif self.mode == 'ep_entropy':
            total_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'total_entropy')
            al_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'al_entropy')
            ep_entropy = total_entropy - al_entropy
            x = ep_entropy

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
        n_classes = 2
        n_samples = batch
        device = x.device
        dropout_predictions = torch.empty((0, n_samples, n_classes), device=device)
        
        for i in range(forward_passes):
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

        if mode == 'al_entropy':
            return entropy[:, None]
        
        if mode == 'total_entropy':
            return self.total_entropy(mean)
        
        if mode == "mean":
            return mean
        

def train(model: torch.nn.Module, 
          device: torch.device, 
          train_loader: DataLoader, 
          optimizer: optim.Optimizer, 
          epoch: int) -> None:
    
    model.train() 
    total = 0      
    correct = 0     
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target) 
        loss.backward()
        optimizer.step()
        
        # Calculate number of correctly classified samples in the current batch
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} '
                  f'Loss: {loss.item():.6f} Correct: {correct}/{total}')

def create_shap_plots(model: torch.nn.Module, 
                      background: Tensor, 
                      test_images: Tensor, 
                      mode: str,
                      class1: str,
                      class2: str) -> None:
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

    plt.ioff()

    if mode in ['al_entropy', 'total_entropy', 'ep_entropy']:
        shap_entropy_numpy = np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2)
        test_entropy_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_entropy_numpy, -test_entropy_numpy, show=False)
        fig = plt.gcf()
        fig.savefig(f"../figs/{class1}_{class2}/{mode}.png")
        plt.close(fig)

    elif mode in ['mean', 'point']:
        # transformed_shap_values = []
        # shap_values_transposed = np.transpose(shap_values, (0, 4, 1, 2, 3))

        # # Iterate over each model output and reshape accordingly
        # for i in range(shap_values_transposed.shape[1]):
        #     shap_for_output = shap_values_transposed[:, i, :, :, :]
        #     shap_for_output = np.reshape(shap_for_output, (shap_for_output.shape[0], 28, 28, 1))
        #     transformed_shap_values.append(shap_for_output)
        # test_numpy = np.swapaxes(np.swapaxes(test_images_cpu.numpy(), 1, -1), 1, 2)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images_cpu.numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy, show=False)
        fig = plt.gcf()
        fig.savefig(f"../figs/{class1}_{class2}/{mode}.png")
        plt.close(fig)

    plt.ion()


def create_dataset(x: int = 1, y: int = 7) -> tuple:
    # Create and return dataset
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
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

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
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=True)
    path = "../models/MCDropout" + str(x) + str(y) + ".pth"
    return train_loader, test_loader, path


def main(args):
    device = torch.device(args.device)
    train_loader, test_loader, path = create_dataset(args.class1, args.class2)
    
    # Initialize and train/test model based on specified mode
    trainPointModel = Net(mode="point").to(device)
    optimizer = optim.SGD(trainPointModel.parameters(), lr=0.01, momentum=0.5)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:2000].to(device)
    test_images = images[0:5]    

    if args.train_mode == 'train':
        for epoch in range(1, args.num_epochs + 1):
            train(trainPointModel, device, train_loader, optimizer, epoch)
            #test(meanModel, device, test_loader)
        torch.save(trainPointModel.state_dict(), path)
        print(f"Model saved at {path}")

    if args.train_mode == 'test':
        if args.inference_mode == 'point':
            #Initialise model(s)
            pointModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            pointModel.load_state_dict(torch.load(path))
            out = pointModel(test_images.to(device))

            create_shap_plots(pointModel, background, test_images, args.inference_mode, args.class1, args.class2)

        elif args.inference_mode == 'mean':
            #Initialise model(s)
            meanModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            meanModel.load_state_dict(torch.load(path))
            out = meanModel(test_images.to(device))

            create_shap_plots(meanModel, background, test_images, args.inference_mode, args.class1, args.class2)

        elif args.inference_mode == 'al_entropy':
            #Initialise model(s)
            alEntropyModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            alEntropyModel.load_state_dict(torch.load(path))

            entropies = alEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(alEntropyModel, background, test_images, args.inference_mode, args.class1, args.class2)

        elif args.inference_mode == 'total_entropy':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            totalEntropyModel.load_state_dict(torch.load(path))

            entropies = totalEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(totalEntropyModel, background, test_images, args.inference_mode, args.class1, args.class2)

        elif args.inference_mode == 'all':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=args.forward_passes, mode='total_entropy').to(device)
            totalEntropyModel.load_state_dict(torch.load(path))
            alEntropyModel = Net(forward_passes=args.forward_passes, mode='al_entropy').to(device)
            alEntropyModel.load_state_dict(torch.load(path))
            epEntropyModel = Net(forward_passes=args.forward_passes, mode='ep_entropy').to(device)
            epEntropyModel.load_state_dict(torch.load(path))
            pointModel = Net(forward_passes=args.forward_passes, mode='point').to(device)
            pointModel.load_state_dict(torch.load(path))

            #Filter subsample for high entropy
            entropies = totalEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)
            test_images.to(device)
            print('breakdown')
            create_shap_plots(totalEntropyModel, background, test_images, 'total_entropy', args.class1, args.class2)
            create_shap_plots(alEntropyModel, background, test_images, 'al_entropy', args.class1, args.class2)
            create_shap_plots(epEntropyModel, background, test_images, 'ep_entropy', args.class1, args.class2)
            create_shap_plots(pointModel, background, test_images, 'point', args.class1, args.class2)
        else:
            print("Invalid inference mode selected!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeurIPS Paper Code')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training/testing.')
    parser.add_argument('--inference_mode', type=str, default='all', help='Mode for inference.')
    parser.add_argument('--train_mode', type=str, default='test', help='Mode for training.')
    parser.add_argument('--forward_passes', type=int, default=50, help='Number of forward passes.')
    parser.add_argument('--class1', type=int, default=4, help='Class 1 for the dataset.')
    parser.add_argument('--class2', type=int, default=9, help='Class 2 for the dataset.')

    args = parser.parse_args()
    main(args)
