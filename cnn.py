import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
cpu = torch.device("cpu")
gpu = torch.device('cuda')
class dataset(Dataset):
    def __init__(self, dataPath, training=True):
        self.dataPath = dataPath
        df = pd.read_csv(dataPath)
        if training:
            self.x = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            for i in self.x:
                i = i/255
            #print(self.x)
            self.y = torch.from_numpy(df.iloc[:,0].values)
        else:
            self.x = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = None
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        x = self.conv(x)
        m = torch.nn.AdaptiveAvgPool2d((1,1))
        x = m(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(dim=1, input=x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             #nn.Conv2d(32, 10, kernel_size=3, padding=1),
#         )
#         self.fc = nn.Linear(7*7*32, 10) 
#     def forward(self, x):
#         x = self.conv(x)
# #         m = torch.nn.AdaptiveAvgPool2d((1,1))
# #         x = m(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc(x))
#         x = F.softmax(dim=1, input=x)
#         return x

batch_size = 100
epoch = 1000
train = dataset("./train.csv")
test = dataset("./test.csv", training=False)

train_loader = Data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = Data.DataLoader(test, batch_size = batch_size, shuffle = False)

model = Net().to(gpu)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
besti = 0
btval = 9999
for it in range(epoch):
    tloss = 0
    tval = 0
    for i, (img, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.reshape(-1, 1, 28, 28)
        #print(img.shape)
#         if i <300:
        output = model(img.to(gpu, dtype=torch.float32))
        #print(torch.argmax(output,dim=1)[0], labels[0])
        loss = criterion(output, labels.to(gpu))
        loss.backward()
        tloss += loss.data
        optimizer.step()
#         else:
#             output = model(img.to(gpu, dtype=torch.float32))
#             tval += criterion(output, labels.to(gpu)).data
    print("[%3d / 100] , [%.3f]"% (it, tloss/300))#, tval/120))
    if tloss < btval:
        besti = it
        btval = tloss
        
        torch.save(model, './mnist_cnn'+str(it)+'.pt')

#test
model = torch.load('./mnist_cnn'+str(besti)+'.pt')
model.eval()
test_pred = torch.LongTensor()
for i, x in enumerate(test_loader):
    x = x.reshape(-1, 1, 28, 28)
    pred = model(x.to(gpu, dtype=torch.float32))
    pred = torch.argmax(pred, dim=1)
    test_pred = torch.cat((test_pred, pred.to(cpu)), dim=0)
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_pred)+1)[:,None], test_pred.numpy()], 
                      columns=['ImageId', 'Label'])

out_df.head()
out_df.to_csv('submission_gap.csv', index=False)