import os
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Define your model architecture here
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # TODO: Define the forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # TODO: Implement the training loop here
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{} \t Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break

def test_epoch(model, device, data_loader):
    # TODO: Implement the testing loop here
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    # TODO: Define your command line arguments here
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
    parser.add_argument('--resume', action='store_true', default=False,
                    help='start training from the latest checkpoint saved')
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # TODO: Load the MNIST dataset for training and testing
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2,**kwargs)
    model = Net().to(device)
    # TODO: Add a way to load the model checkpoint if 'resume' argument is True
    past_epochs = 0
    if args.resume:
        checkpoints = glob.glob('/workspace/mnist_cnn_epoch_*.pth')  # Get a list of all checkpoint files
        if checkpoints:
            past_epochs = len(checkpoints)
            checkpoints.sort()  # Sort the checkpoints based on file names
            latest_checkpoint = checkpoints[-1]  # Select the latest checkpoint
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint)
            # Additional code to resume training or use the loaded model
        else:
            print("No checkpoint found.")
    

    # TODO: Choose and define the optimizer here
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # TODO: Implement the training and testing cycles
    for epoch in range(1, args.epochs + 1):
        epoch_save_num = past_epochs + epoch # adding past epochs to avoid overwriting older checkpoints in case of resume
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        test_epoch(model, device, test_loader)
        torch.save(model.state_dict(), f"/workspace/mnist_cnn_epoch_{str(epoch_save_num)}.pth")

    # Hint: Save the model after each epoch

if __name__ == "__main__":
    main()
