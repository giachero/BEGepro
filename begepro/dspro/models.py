import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

###########
# DATASET #
###########

# Dataset used for training and validation
class Dataset(torch.utils.data.Dataset):
  def __init__(self, coll):
    self.coll = coll
      
  def __len__(self):
    return self.coll.n_trace
  
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.to_list()

    # Normalization of the signals
    normalized_trace = self.coll.get_traces()[idx]
    normalized_trace = normalized_trace - min(normalized_trace)
    normalized_trace = normalized_trace / max(normalized_trace)
    normalized_trace = torch.tensor(normalized_trace)

    # The label is obtained from the energy. Label 0 for SSE and 1 for MSE
    energy = self.coll.get_energies()[idx]
    label = np.array([0]) if ((energy > 1590) & (energy < 1596)) | ((energy > 2250) & (energy <2375)) else np.array([1])
    return torch.Tensor(normalized_trace), torch.Tensor(label)

#########
# MODEL #
#########
  
class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 50 , 5)
        self.norm = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 10)
        self.conv3 = nn.Conv1d(32, 64, 10)
        self.fc1 = nn.Linear(64, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)

        self.criterion = nn.CrossEntropyLoss() #Cross entropy
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)#, momentum=0.9)
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 1300*5, eta_min = 1e-6)

    def forward(self, x):
        x = self.pool(self.norm(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()
        return loss.item(), outputs