import numpy as np
import torch


class SimpleReplayBufferIros(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        np.random.seed(101)
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

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
        np.random.seed(101)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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

class CostReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        np.random.seed(101)
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
        np.random.seed(101)
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
        np.random.seed(101)
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