# Reinforcement Learning
# Fabrice LAURI
#
# Q-Iteration
#
# Reinforcement Learning
# Fabrice LAURI
#
# MDPs from work session #1-#2
#


import numpy as np
import copy
from itertools import *


directions={0:np.array((0,1)),1:np.array((0,-1)),2:np.array((-1,0)),3:np.array((1,0))}
direction_names=['r','l','u','d']


class StarGathering_template:
    def __init__(self, gamma=0.9, positive_reward=1):
        self.Gamma=gamma
        self.PR=positive_reward
        self.buildT()


    # Set of states
    def stateSpace(self):
        for i in range(3):
            for j in range(3):
                yield (i,j)

    # Set of actions per state
    # (same set for all states)
    def actionSpace(self,s=None):
        return range(4)


    def actionName(self,a):
        return direction_names[a]

    def actionNames(self,action_list):
        l=''
        n=len(action_list)
        for k,a in enumerate(action_list):
            if k>0 and k<n:
                l+=','
            l+=self.actionName(a)
        return l


    def buildT(self):
        # Define the dictionary self.T (key is (s,a))
        # that specifies the next state reached 
        # after doing action a in state s
        self.Tm=dict()
        pass


    def displayT(self):
        for s in self.stateSpace():
            for a in self.actionSpace(s):
                print('{},{}: {}'.format(s,self.actionName(a),self.T(s,a)))


    def T(self,s,a):
        return self.Tm[(s,a)]

    def R(self,s,a):
        # Reward function
        return 0


#
# Q-Iteration
#
class QIteration():
    def __init__(self,mdp):
        self.MDP=mdp
        self.Q=dict()
        pass
    
    def run(self,N=-1,precision=1e-13):
        # apply N iterations of Q-Iteration
        # N=-1 means executing until convergence
        l=0
        while(l!=N):
            oldQ=copy.deepcopy(self.Q)

            # Here is the updating of the function Q

            # Stops until convergence, when the precision has been reached for each tuple (s,a)
            if N==-1 and (abs(np.array(list(oldQ.values()))-np.array(list(self.Q.values())))<precision).all():
                print('QIteration has stopped after',l,'iterations!')
                break


class Policy:
    def __init__(self,mdp,Q):
        self.MDP=mdp
        self.Q=Q

    def bestActions(self,s):
        l=self.Q[s]
        return np.argwhere(l==np.amax(l)).flatten().tolist()

    def displayPi(self):
        print('Pi:')
        for s in self.MDP.stateSpace():
            print('{}: {}'.format(s,self.MDP.actionNames(self.bestActions(s))))


def launch(mdp_class,RL_tec,N=-1,precision=1e-13):
    mdp=mdp_class

    # Application of Q-Iteration to this MDP
    Qitr=RL_tec(mdp)
    Qitr.run(N=N,precision=precision)
    #
    # Compute the policy and display it
    pi=Policy(mdp,Qitr.Q)
    pi.displayPi()


# Example of execution
# (this program will not execute properly, since the class StarGathering
#  has not been defined and the class QIteration is incomplete...)
launch(StarGathering(positive_reward=100),QIteration)
