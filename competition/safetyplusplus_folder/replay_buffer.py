import numpy as np
import torch


class SimpleReplayBufferIros(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.phrase=4
        self.one_phrase_max_size= int(max_size/self.phrase)
        self.ptr = [0,0,0,0]

        self.size = [0,0,0,0]
        
        self.state = np.zeros(( self.phrase,self.one_phrase_max_size, state_dim))
        self.action = np.zeros(( self.phrase,self.one_phrase_max_size, action_dim))
        self.next_state = np.zeros(( self.phrase,self.one_phrase_max_size, state_dim))
        self.reward = np.zeros(( self.phrase,self.one_phrase_max_size, 1))
        self.not_done = np.zeros(( self.phrase,self.one_phrase_max_size, 1))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def add(self,phrase, state, action, next_state, reward, done):
        ptr=self.ptr[phrase]
        self.state[phrase][ptr] = state
        self.action[phrase][ptr] = action
        self.next_state[phrase][ptr] = next_state
        self.reward[phrase][ptr] = reward
        self.not_done[phrase][ptr] = 1. - done

        self.ptr[phrase] = (self.ptr[phrase] + 1) % self.one_phrase_max_size
        self.size[phrase] = min(self.size[phrase] + 1, self.one_phrase_max_size)


    def sample(self, batch_size):
        # inverse sample  , give the end phrase the top priotity
        size_3= 64 if self.size[3] > batch_size/4  else self.size[3]
        size_2= 64 if self.size[2] > batch_size/4  else self.size[2]
        size_1= 64 if self.size[1] > batch_size/4  else self.size[1]
        size_0= batch_size -size_1 - size_2-size_3

        ind0 = np.random.randint(0, self.size[0], size=size_0)
        ind1 = np.random.randint(0, self.size[1], size=size_1)
        ind2 = np.random.randint(0, self.size[2], size=size_2)
        ind3 = np.random.randint(0, self.size[3], size=size_3)

        if size_0 !=0:
            state=self.state[0][ind0]
            action=self.action[0][ind0]
            next_state=self.next_state[0][ind0]
            reward=self.reward[0][ind0]
            not_done=self.not_done[0][ind0]
        if size_1!=0:
            state=np.vstack((state,self.state[1][ind1]))
            action=np.vstack((action,self.action[1][ind1]))
            next_state=np.vstack((next_state,self.next_state[1][ind1]))
            reward=np.vstack((reward,self.reward[1][ind1]))
            not_done=np.vstack((not_done,self.not_done[1][ind1]))
        if size_2!=0:
            state=np.vstack((state,self.state[2][ind2]))
            action=np.vstack((action,self.action[2][ind2]))
            next_state=np.vstack((next_state,self.next_state[2][ind2]))
            reward=np.vstack((reward,self.reward[2][ind2]))
            not_done=np.vstack((not_done,self.not_done[2][ind2]))
        if size_3!=0:
            state=np.vstack((state,self.state[3][ind3]))
            action=np.vstack((action,self.action[3][ind3]))
            next_state=np.vstack((next_state,self.next_state[3][ind3]))
            reward=np.vstack((reward,self.reward[3][ind3]))
            not_done=np.vstack((not_done,self.not_done[3][ind3])) 
        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(not_done).to(self.device),
        )

class SimpleReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class IrosReplayBuffer(object):
    def __init__(self, global_state_dim,local_state_shape ,action_dim, max_size=int(1e5),load=True):
        self.max_size = max_size
        
        z=local_state_shape[0]
        x=local_state_shape[1]
        y=local_state_shape[2]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_state = torch.zeros((max_size, global_state_dim)).to(self.device)
        self.local_state = torch.zeros((max_size, z,x,y)).to(self.device)
        self.action = torch.zeros((max_size, action_dim)).to(self.device)
        self.next_global_state = torch.zeros((max_size, global_state_dim)).to(self.device)
        self.next_local_state = torch.zeros((max_size, z,x,y)).to(self.device)
        self.reward = torch.zeros((max_size, 1)).to(self.device)
        self.not_done = torch.zeros((max_size, 1)).to(self.device)
        self.load=load
        self.pretrain_size=int(1e4)
        if self.load :
            self.load_buffer()
            self.ptr = self.pretrain_size
            self.size = self.pretrain_size
        else:
            self.ptr = 0
            self.size = 0

    def add(self, global_state,local_state, action, next_global_state,next_local_state, reward, done):
        
        self.global_state[self.ptr] = torch.FloatTensor(global_state)
        self.local_state[self.ptr] = torch.FloatTensor(local_state)
        self.action[self.ptr] = torch.FloatTensor(action) 
        self.next_global_state[self.ptr] = torch.FloatTensor(next_global_state)
        self.next_local_state[self.ptr] = torch.FloatTensor(next_local_state)
        self.reward[self.ptr] = torch.FloatTensor(np.array([reward]))
        self.not_done[self.ptr] = torch.FloatTensor(np.array([1. - done]))
        if  self.load:
            self.ptr = (self.ptr + 1) if self.ptr + 1 !=self.max_size else self.pretrain_size
        else:
            self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        # import pdb;pdb.set_trace()
        # print(ind)
        return (self.global_state[ind],self.local_state[ind],self.action[ind],self.next_global_state[ind],self.next_local_state[ind],self.reward[ind],self.not_done[ind])
    
    def write(self):
        np.save("buffer_npy/global_state.npy",np.array(self.global_state[0:self.pretrain_size].cpu()))
        np.save("buffer_npy/local_state.npy",np.array(self.local_state[0:self.pretrain_size].cpu()))
        np.save("buffer_npy/action.npy",np.array(self.action[0:self.pretrain_size].cpu()))
        np.save("buffer_npy/next_global_state.npy",np.array(self.next_global_state[0:self.pretrain_size].cpu()))
        np.save("buffer_npy/next_local_state.npy",np.array(self.next_local_state[0:self.pretrain_size].cpu()))
        np.save("buffer_npy/reward.npy",np.array(self.reward[0:self.pretrain_size].cpu()))
        np.save("buffer_npy/not_done.npy",np.array(self.not_done[0:self.pretrain_size].cpu()))

    def load_buffer(self):
        expert_global_state=torch.FloatTensor(np.load("buffer_npy/global_state.npy"))
        expert_local_state=torch.FloatTensor(np.load("buffer_npy/local_state.npy"))
        expert_action=torch.FloatTensor(np.load("buffer_npy/action.npy"))
        expert_next_global_state=torch.FloatTensor(np.load("buffer_npy/next_global_state.npy"))
        expert_next_local_state=torch.FloatTensor(np.load("buffer_npy/next_local_state.npy"))
        expert_reward=torch.FloatTensor(np.load("buffer_npy/reward.npy"))
        expert_not_done=torch.FloatTensor(np.load("buffer_npy/not_done.npy"))
        self.global_state[0:self.pretrain_size] = expert_global_state
        self.local_state[0:self.pretrain_size] = expert_local_state
        self.action[0:self.pretrain_size] = expert_action
        self.next_global_state[0:self.pretrain_size] = expert_next_global_state
        self.next_local_state[0:self.pretrain_size] = expert_next_local_state
        self.reward[0:self.pretrain_size] = expert_reward
        self.not_done[0:self.pretrain_size] = expert_not_done
        print("load ok")
    

class CostReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, cost, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class RecReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.raw_action = np.zeros((max_size, action_dim))
        self.rec_action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, raw_action, rec_action, next_state, reward, cost, done):
        self.state[self.ptr] = state
        self.raw_action[self.ptr] = raw_action
        self.rec_action[self.ptr] = rec_action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.raw_action[ind]).to(self.device),
            torch.FloatTensor(self.rec_action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class SafetyLayerReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))
        self.prev_cost = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, cost, prev_cost,done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.prev_cost[self.ptr] = prev_cost
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
            torch.FloatTensor(self.prev_cost[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )