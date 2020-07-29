import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        #model Layers
        self.conv1 = nn.Conv2d(1,16,2) #28-2+1 = 27 
        self.conv2 = nn.Conv2d(16,32,2)#27-2+1 = 26
        self.conv3 = nn.Conv2d(32,64,2)#26-2+1 = 25
        self.conv4 = nn.Conv2d(64,128,2)#25-2+1 = 24
        self.max = nn.MaxPool2d(2,2)
        self.linear = nn.Linear(12*12*128,10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.max(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.linear(x))
        x = F.log_softmax(x, dim=1)
        
        return x
        
        