import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import MammothBackbone


class WM_Vision(MammothBackbone):
    def __init__(self, nclasses=4):
        print("Number of classes:", nclasses)
        super(MammothBackbone, self).__init__()
        self.rawCNN = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16, stride=2),
            nn.ReLU(),
            #nn.LayerNorm(32),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.LayerNorm(128),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.LayerNorm(64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.bn = nn.LayerNorm([1,32,14]) #nn.BatchNorm1d(32)
        self.others = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
            )
        
        
        self.fc1raw = nn.LazyLinear(64 + 4) #size of one hot directional vector
        
        self.sparse = nn.Dropout(0.2) #nn.Identity() # nn.Dropout(0.2)
        self.out_dim = 32
        self.fc2raw = nn.Linear(64+4, 32)
        
        
#        self.out = nn.Linear(32, nclasses)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        self.forward([torch.tensor(np.zeros(1000)).reshape(1,1000).float(), torch.zeros(4)])


    @property
    def feature_dim(self):
        return 32



    def forward(self, raw, returnt = "out"):
        direction = raw[1]
        raw = raw[0]
        raw = raw.unsqueeze(dim = 1)
        
        
        
        raw = self.rawCNN(raw)
        #print(raw.shape)
        #raise Exception
        bn_raw = self.bn(raw)
        raw = self.others(bn_raw)
        
        #print(raw.shape)
        
        raw = raw.view(raw.shape[0], -1).reshape(96)
        
        #print(raw.reshape(96).shape)
        
        raw = F.relu(self.fc1raw(torch.cat((raw, direction))))
        
        raw = self.sparse(raw)
        
        features = raw.view(raw.size[0], -1)
        
        if returnt == "features":
            return features
        #added another layer
        out = F.relu(self.fc2raw(raw))
        if returnt == "out":
            return out
        if returnt == "all":
            return (features, out)
#        x = self.out(raw)

def wm_vision(nclasses = 4, nf: int = 64):
    return WM_Vision()
        
        