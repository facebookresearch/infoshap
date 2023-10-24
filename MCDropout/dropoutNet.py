import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import shap

class Net(nn.Module):
    def __init__(self, forward_passes: int = 20, mode: str = 'mean'):
        super(Net, self).__init__()
        # Initialize variables
        self.forward_passes = forward_passes
        self.mode = mode

        # Define model layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'point':
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)

        elif self.mode == 'total_entropy':
            x = self.conv_layers(x)
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
        entropy = -torch.sum(x * torch.log(x), dim=-1)
        return entropy[:,None]

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_monte_carlo_predictions(self, x, batch, forward_passes, mode, device=torch.device('cuda')):
        n_classes = 2
        n_samples = batch
        dropout_predictions = torch.empty((0, n_samples, n_classes), device=device)
        for i in range(forward_passes):
            predictions = torch.empty((0, n_classes), device=device)
            self.enable_dropout()
            conv_out = self.conv_layers(x)
            reshape_out = conv_out.view(-1, 320)
            predictions = self.fc_layers(reshape_out)

            dropout_predictions = torch.cat((dropout_predictions, predictions.unsqueeze(0)), dim=0)
            

        # Calculating mean across multiple MCD forward passes 
        mean = dropout_predictions.mean(dim=0) # shape (n_samples, n_classes)

        # Calculating entropy across multiple MCD forward passes 
        entropy = -torch.sum(dropout_predictions * torch.log(dropout_predictions), dim=-1)
        entropy = entropy.mean(dim=0)
        
        if mode =='mean':
            return mean
        
        if mode == 'ep_entropy':
            return entropy[:,None]
        

def train(model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer, epoch: int):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def create_shap_plots(model: nn.Module, background: torch.Tensor, 
                      test_images: torch.Tensor, mode: str):
    # Generate SHAP plots
    explainer = shap.DeepExplainer(model, background)
    shap_values = np.asarray(explainer.shap_values(test_images))
    #Create shap plots
    if mode == 'al_entropy' or mode == 'total_entropy' or mode == 'ep_entropy': 
        test_images_cpu = test_images.to(torch.device('cpu'))
        shap_entropy_numpy = np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2)
        test_entropy_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_entropy_numpy, -test_entropy_numpy)

    elif mode == 'mean' or mode == 'point':
        test_images_cpu = test_images.to(torch.device('cpu'))
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images_cpu.numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)


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
    path = r"..\models\MCDropout" + str(x) + str(y) + ".pth"
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

    if args.train_mode == 'test':
        if args.inference_mode == 'point':
            #Initialise model(s)
            pointModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            pointModel.load_state_dict(torch.load(path))
            out = pointModel(test_images.to(device))

            create_shap_plots(pointModel, background, test_images, args.inference_mode)

        elif args.inference_mode == 'mean':
            #Initialise model(s)
            meanModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            meanModel.load_state_dict(torch.load(path))
            out = meanModel(test_images.to(device))

            create_shap_plots(meanModel, background, test_images, args.inference_mode)

        elif args.inference_mode == 'al_entropy':
            #Initialise model(s)
            alEntropyModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            alEntropyModel.load_state_dict(torch.load(path))

            entropies = alEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(alEntropyModel, background, test_images, args.inference_mode)

        elif args.inference_mode == 'total_entropy':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=args.forward_passes, mode=args.inference_mode).to(device)
            totalEntropyModel.load_state_dict(torch.load(path))

            entropies = totalEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(totalEntropyModel, background, test_images, args.inference_mode)

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
            #max_entropy = torch.topk(entropies.flatten(), 5)
            #indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            #test_images = images[indices_cpu].to(device)
            test_images.to(device)
            print('breakdown')
            create_shap_plots(totalEntropyModel, background, test_images, 'total_entropy')
            create_shap_plots(alEntropyModel, background, test_images, 'al_entropy')
            create_shap_plots(epEntropyModel, background, test_images, 'ep_entropy')
            create_shap_plots(pointModel, background, test_images, 'point')
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
