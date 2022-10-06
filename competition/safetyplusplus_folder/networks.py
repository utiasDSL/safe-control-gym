import torch
import torch.nn as nn
import torch.nn.functional as F

num=256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        # (5,23,23)
        self.conv1=nn.Conv2d(in_channels,16,3) # 
        self.conv2=nn.Conv2d(16,32,3) # 
    def forward(self,x):
        # import pdb;pdb.set_trace()
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2) # (16,10,10)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2,2)
        out = torch.mean(out,dim=1,keepdim=True)
        out = out.view(in_size,-1) # 扁平化flat然后传入全连接层
        # import pdb;pdb.set_trace()
        return out


class Actor(nn.Module):
    def __init__(self, state_dim,in_channels,out_channels, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv1=nn.Conv2d(in_channels,16,3) # 
        self.conv2=nn.Conv2d(16,32,3) # 

        self.l1 = nn.Linear(state_dim+out_channels, num)
        self.l2 = nn.Linear(num, num)
        self.l3 = nn.Linear(num, action_dim)
        
        self.max_action = max_action

    
    def forward(self, all_state):
        global_state=all_state[0]
        local_state=all_state[1]

        
        in_size = local_state.size(0)
        out_local = self.conv1(local_state)
        out_local = F.relu(out_local)
        out_local = F.max_pool2d(out_local, 2, 2) 
        out_local = self.conv2(out_local)
        out_local = F.relu(out_local)
        out_local = F.max_pool2d(out_local,2,2)
        out_local = torch.mean(out_local,dim=1,keepdim=True)
        out_local = out_local.view(in_size,-1) # 扁平化flat然后传入全连接层
        
        all_feature=torch.concat([global_state,out_local],1)

        out = F.relu(self.l1(all_feature))
        out = F.relu(self.l2(out))
        return self.max_action * torch.tanh(self.l3(out))


class C_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(C_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, num)
        self.l2 = nn.Linear(num, num)
        self.l3 = nn.Linear(num, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class Critic(nn.Module):
    def __init__(self, state_dim,in_channels,out_channels, action_dim):
        super(Critic, self).__init__()

        # CNN
        self.conv1=nn.Conv2d(in_channels,16,3) # 
        self.conv2=nn.Conv2d(16,32,3) #
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + out_channels, num)
        self.l2 = nn.Linear(num, num)
        self.l3 = nn.Linear(num, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + out_channels , num)
        self.l5 = nn.Linear(num, num)
        self.l6 = nn.Linear(num, 1)


    def forward(self, all_state, action):
        global_state=all_state[0]
        local_state=all_state[1]
        in_size = local_state.size(0)
        out_local = self.conv1(local_state)
        out_local = F.relu(out_local)
        out_local = F.max_pool2d(out_local, 2, 2) # (16,10,10)
        out_local = self.conv2(out_local)
        out_local = F.relu(out_local)
        out_local = F.max_pool2d(out_local,2,2)
        out_local = torch.mean(out_local,dim=1,keepdim=True)
        out_local = out_local.view(in_size,-1) # 扁平化flat然后传入全连接层
        # import pdb;pdb.set_trace()
        all_feature=torch.concat([global_state,out_local],1)

        sa = torch.cat([all_feature, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, all_state, action):
        global_state=all_state[0]
        local_state=all_state[1]
        in_size = local_state.size(0)
        out_local = self.conv1(local_state)
        out_local = F.relu(out_local)
        out_local = F.max_pool2d(out_local, 2, 2) # (16,10,10)
        out_local = self.conv2(out_local)
        out_local = F.relu(out_local)
        out_local = F.max_pool2d(out_local,2,2)
        out_local = torch.mean(out_local,dim=1,keepdim=True)
        out_local = out_local.view(in_size,-1) # 扁平化flat然后传入全连接层
        # import pdb;pdb.set_trace()
        all_feature=torch.concat([global_state,out_local],1)

        sa = torch.cat([all_feature, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# predict state-wise \lamda(s)
class MultiplerNet(nn.Module):
    def __init__(self, state_dim):
        super(MultiplerNet, self).__init__()

        self.l1 = nn.Linear(state_dim, num)
        self.l2 = nn.Linear(num, num)
        self.l3 = nn.Linear(num, 1)

        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        #return F.relu(self.l3(a))
        return F.softplus(self.l3(a)) # lagrangian multipliers can not be negative

# import numpy as np 
# encoder=Encoder(5)
# a=np.zeros([1,5,23,23])
# encoder=encoder.float()
# out=encoder(torch.tensor(a).float())
