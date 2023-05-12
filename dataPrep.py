import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Settings:
img_size = 64
num_class = 10
test_size = 0.15

def get_dataset():
    # Getting all data from data path:
    try:
        data = np.load('Data/X.npy')
        labels = np.load('Data/Y.npy')
    except:
        print("Error loading data")
    data, data_test, labels, labels_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    
    return data, data_test, labels, labels_test

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Input is grayscale, so 1 input channel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(30*30*64, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    data, data_test, labels, labels_test = get_dataset()
    data = data.reshape(-1, 1, img_size, img_size)  # Reshape the data to match the input size of the model
    data_test = data_test.reshape(-1, 1, img_size, img_size)

    # Convert to PyTorch tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.from_numpy(data).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)
    data_test = torch.from_numpy(data_test).float().to(device)
    labels_test = torch.from_numpy(labels_test).long().to(device)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model.train()
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.max(labels, 1)[1])  # Convert one-hot to index
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        output = model(data_test)
        predicted = torch.max(output.data, 1)[1]
        correct = (predicted == torch.max(labels_test, 1)[1]).sum().item()
        total = labels_test.size(0)
        print(f"Test Accuracy: {correct / total * 100}%")
