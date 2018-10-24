import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from .cuda_setup import device




class DQNImage(nn.Module):

    def __init__(self, env):
        super(DQNImage, self).__init__()

        self.INPUT_TYPE = "IMAGE2D"

        #Using 3 as input because we are taking raw pixel as input
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        #print(x)
        return x



class DQNStateSpace(nn.Module):
    def __init__(self, env):
        super(DQNStateSpace, self).__init__()

        self.INPUT_TYPE = "STATESPACE"

        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)


    def forward(self, x):

        x = F.relu(F.dropout(self.l1(x), p=0.6))
        x = F.softmax(self.l2(x), dim=-1)
        """
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        """
        #print(x)
        return x


"""
COMING SOON: Implemention of full smart embedding tabular network for structured and typed inputs

def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def get_embedding(ni,nf):
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb

class DQNStateSpace2(nn.Module):

    def __init__(self, env, emb_szs=[], n_cont=5):
        super(DQNStateSpace2, self).__init__()

        #env.observation_space.shape

        def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int],
                ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True):

        emb_szs = [] #ListSizes
        n_cont = 4 #int
        n_cont = env.observation_space.shape[0]
        out_sz = env.action_space.n
        layers = [3] #Collection[int]
        ps = None #Collection[float]=None
        emb_drop = 0. #float=0.
        y_range = None #OptRange=None
        use_bn = True #bool=True

        self.embeds = nn.ModuleList([get_embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        print("actns")
        print(actns)
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        self.layers = nn.Sequential(*layers)

        self.head = nn.Linear(448, env.action_space.n)

    def forward(self, x):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)

        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * F.sigmoid(x) + self.y_range[0]

        x = self.head(x.view(x.size(0), -1))
        #return x.squeeze()
        return x
"""
