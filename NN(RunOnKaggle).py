import yfinance as yf
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

index = "^JKLQ45"

df = yf.download(tickers=[index],period='max')

ordinal_patterns = list(itertools.permutations([1,2,3,4]))
normed_ordinal_patterns = [np.array(pattern)*25 for pattern in ordinal_patterns]

adj_close = df['Adj Close']
list_of_adj_close = []
for i in range(0,4):
    x = adj_close.shift(i)
    list_of_adj_close.append(x)
four_shifts = pd.concat(list_of_adj_close,axis=1).dropna()
four_shifts.columns = ["adj0","adj1","adj2","adj3"]

def process_row(row):
    array_values = []
    for i in range(0,4):
        array_values.append(row[f"adj{i}"])
    return_array = np.array(array_values)
    return_array = (return_array-return_array.min())/(return_array.max()-return_array.min())*100
    return return_array

four_shifts['arrays'] = four_shifts.apply(process_row,axis=1)


def diff_with_ord_patt(row):
    array = row['arrays']
    differences = []
    for pattern in normed_ordinal_patterns:
        differences.append((pattern-array).mean())
    return np.array(differences)

four_shifts['diff_with_ord_patt'] = four_shifts.apply(diff_with_ord_patt,axis=1)

for i in range(0,23):
    four_shifts[f'diff_with_ord_patt_{i+1}'] = four_shifts['diff_with_ord_patt'].shift(i+1)

four_shifts = four_shifts.dropna()

def gridify(row):
    grid = []
    grid.append(row['diff_with_ord_patt'])
    for i in range(0,23):
        grid.append(row[f'diff_with_ord_patt_{i+1}'])
    return np.array(grid)

grids = four_shifts.apply(gridify,axis=1)
next_day_ord_patt = four_shifts['diff_with_ord_patt'].shift(-1)
grids = grids.iloc[:-1]
next_day_ord_patt = next_day_ord_patt.iloc[:-1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


grids = np.stack(grids.values)
next_day_ord_patt = np.stack(next_day_ord_patt.values)

X_train, X_test, y_train, y_test = train_test_split(grids,next_day_ord_patt,test_size=0.33,random_state=42)


X_train = torch.tensor(X_train,device=device,dtype=torch.float32)
X_test = torch.tensor(X_test,device=device,dtype=torch.float32)
y_train = torch.tensor(y_train,device=device,dtype=torch.float32)
y_test = torch.tensor(y_test,device=device,dtype=torch.float32)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #24 x 24
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=3)
        #8 x 8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        #4 x (4 x 4)
        self.fc1 = nn.Linear(4*4*4, 30)
        self.fc3 = nn.Linear(30,24)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        x = self.avgpool(x)
        x = self.relu(self.conv1(x))
        x = x.view(-1,4*4*4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return x.squeeze(0)

class CustomDataset(Dataset):
    def __init__(self,train_tensor,target_tensor):
        self.train_data = train_tensor
        self.target_data = target_tensor
    def __len__(self):
        return len(self.train_data)
    def __getitem__(self,idx):
        sample = self.train_data[idx]
        label = self.target_data[idx]
        return sample,label

cnn = CNN().to(device)
train_set = CustomDataset(X_train,y_train)
test_set = CustomDataset(X_test,y_test)
train_dataloader = DataLoader(train_set,shuffle=True)
test_dataloader = DataLoader(test_set,shuffle=True)

criterion = nn.L1Loss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)

train_loss_values = []
test_loss_values = []
num_epochs = 1000
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs,labels in train_dataloader:
        outputs = cnn(inputs)
        loss = criterion(outputs,labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    average_loss = epoch_loss/len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    train_loss_values.append(average_loss)
    
    test_loss = 0.0
    for inputs,labels in test_dataloader:
        outputs = cnn(inputs)
        loss = criterion(outputs,labels.float())
        test_loss += loss.item()
    average_loss = test_loss/len(test_dataloader)
    test_loss_values.append(average_loss)

plt.plot(range(1, num_epochs + 1), train_loss_values, marker='o')
plt.plot(range(1, num_epochs+1), test_loss_values, marker='X')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()