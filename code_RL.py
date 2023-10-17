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
import random
from itertools import *

directions = {0: np.array((0, 1)), 1: np.array((0, -1)), 2: np.array((-1, 0)), 3: np.array((1, 0))}
direction_names = ['r', 'l', 'u', 'd']
star_cells = [(0, 0), (2, 1), (1, 2)]


class StarGathering:
    def __init__(self, gamma=0.9, positive_reward=1):
        self.Gamma = gamma
        self.PR = positive_reward
        self.buildT()

    def terminal(self, s):
        return s == (1, 1)

    # Set of states
    def stateSpace(self):
        for i in range(3):
            for j in range(3):
                yield (i, j)

    # Set of actions per state
    # (same set for all states)
    def actionSpace(self, s=None):
        return range(4)

    def actionName(self, a):
        return direction_names[a]

    def actionNames(self, action_list):
        l = ''
        n = len(action_list)
        for k, a in enumerate(action_list):
            if k > 0 and k < n:
                l += ','
            l += self.actionName(a)
        return l

    def buildT(self):
        # Define the dictionary self.T (key (s,a))
        # that specifies the next state reached
        # after doing action a in state s
        self.Tm = dict()
        for s in self.stateSpace():
            for d in directions:
                ns = np.array(s) + directions[d]
                if 0 <= ns[0] < 3 and 0 <= ns[1] < 3:
                    self.Tm[(s, d)] = tuple(ns)
                else:
                    self.Tm[(s, d)] = s
        pass

    def displayT(self):
        for s in self.stateSpace():
            for a in self.actionSpace(s):
                print('{},{}: {}'.format(s, self.actionName(a), self.T(s, a)))

    def T(self, s, a):
        return self.Tm[(s, a)]

    def R(self, s, a):
        # Reward function
        return 0


#
# Q-Iteration
#
class QIteration():
    def __init__(self, mdp):
        self.MDP = mdp
        self.Q = dict()
        nba = 0
        for s in self.MDP.stateSpace():
            nba = max([nba] + list(self.MDP.actionSpace(s)))
        for s in self.MDP.stateSpace():
            self.Q[s] = np.zeros(nba + 1)
        pass

    def run(self, N=-1, precision=1e-13):
        # apply N iterations of Q-Iteration
        # N=-1 means executing until convergence
        l = 0
        while (l != N):
            oldQ = copy.deepcopy(self.Q)

            # Here is the updating of the function Q

            for s in self.MDP.stateSpace():
                if not self.MDP.terminal(s):
                    for a in self.MDP.actionSpace(s):
                        self.Q[s][a] = self.MDP.R(s, a) + self.MDP.Gamma * np.max(self.Q[self.MDP.T(s, a)])

            if N != -1:
                print('Iteration:', l)

                for s in self.MDP.stateSpace():
                    for k, a in enumerate(self.MDP.actionSpace(s)):
                        print(str(round(self.Q[s][a], 2)).ljust(7), end='')
                    print('')

            # Stops until convergence, when the precision has been reached for each tuple (s,a)
            if N == -1 and (abs(np.array(list(oldQ.values())) - np.array(list(self.Q.values()))) < precision).all():
                print('QIteration has stopped after', l, 'iterations!')
                break


#
# Q-Learning
#
class QLearning():

    def __init__(self, mdp):
        self.MDP = mdp
        self.Q = dict()
        nba = 0
        for s in self.MDP.stateSpace():
            nba = max([nba] + list(self.MDP.actionSpace(s)))
        for s in self.MDP.stateSpace():
            self.Q[s] = np.zeros(nba + 1)
        pass

    def bestActions(self, s):
        actions = self.MDP.actionSpace(s)
        if len(actions) > 0:
            values = self.Q[s][actions]
            best_actions = np.argwhere(values == np.amax(values)).flatten().tolist()
            return [actions[k] for k in best_actions]
        else:
            return []

    def EpsilonGreedy(self, s, epsilon):
        r = random.random()
        if r < epsilon:
            return random.choice(self.MDP.actionSpace(s))
        else:
            return random.choice(self.bestActions(s))

    def run(self, N=-1, precision=1e-13, epsilon=0.1, alpha=0.9):
        # apply N episodes of Q-Learning
        # N=-1 means executing until convergence
        S = list(self.MDP.stateSpace())

        l = 0
        while (l != N):
            l += 1
            oldQ = copy.deepcopy(self.Q)

            s = random.choice(S)

            while (True):
                if self.MDP.terminal(s):
                    break

                a = self.EpsilonGreedy(s, epsilon)
                ns = self.MDP.T(s, a)
                r = self.MDP.R(s, a)
                #                print(s,a,'->',ns,' (',r,')',self.Q[s][a])

                newtarget = r + self.MDP.Gamma * np.max(self.Q[ns])
                self.Q[s][a] += alpha * (newtarget - self.Q[s][a])

                s = ns

            if N != -1 and N < 10:
                print('Iteration:', l)

                for s in self.MDP.stateSpace():
                    for k, a in enumerate(self.MDP.actionSpace(s)):
                        print(str(round(self.Q[s][a], 2)).ljust(7), end='')
                    print('')

            if N == -1 and (abs(np.array(list(oldQ.values())) - np.array(list(self.Q.values()))) < precision).all():
                print('QIteration has stopped after', l, 'iterations!')
                break


class Policy:
    def __init__(self, mdp, Q):
        self.MDP = mdp
        self.Q = Q

    def bestActions(self, s):
        l = self.Q[s]
        return np.argwhere(l == np.amax(l)).flatten().tolist()

    def displayPi(self):
        print('Pi:')
        for s in self.MDP.stateSpace():
            print('{}: {}'.format(s, self.MDP.actionNames(self.bestActions(s))))


class StarCollector():

    def __init__(self, gamma=0.9, positive_reward=1, m0=None):
        self.Gamma = gamma
        self.PR = positive_reward
        if m0 is not None:
            # Reachable states are those that can be visited from the stars' locations in m0
            self.ReachableStates = self.reachableStates(m0)


        pass
        if m0 is not None:
            # Reachable states are those that can be visited from the stars' locations in m0
            self.ReachableStates = self.reachableStates(m0)

    def stateSpace(self):
        for m in product([0, 1], repeat=9):
            for i in range(3):
                for j in range(3):
                    yield (m, (i, j))

    def actionSpace(self, s=None):
        return range(4)

    def actionName(self, a):
        return direction_names[a]

    def actionNames(self, action_list):
        l = ''
        n = len(action_list)
        for k, a in enumerate(action_list):
            if k > 0 and k < n:
                l += ','
            l += self.actionName(a)
        return l

    def terminal(self, s):
        _, c = s
        return c == (1, 1)

    def T(self, s, a):
        m, c = s
        m = np.array(m).reshape(3, 3)
        new_c = tuple(np.array(c) + directions[a])
        if new_c[0] < 0 or new_c[0] >= 3 or new_c[1] < 0 or new_c[1] >= 3:
            new_c = c

        new_m = copy.deepcopy(m)
        if m[new_c[1], new_c[0]] == 1:
            new_m[new_c[1], new_c[0]] = 0

        return (tuple(new_m.flatten()), new_c)

    def R(self, s, a):
        # Reward function
        m, _ = s
        m = np.array(m).reshape(3, 3)
        _, nc = self.T(s, a)
        if m[nc[1], nc[0]] == 1:
            return self.PR
        else:
            return 0

    def reachableStates(self, m0):
        coords = []
        for k, v in enumerate(m0):
            if v:
                coords.append(k)
        nb_coords = len(coords)

        l = []
        for p in product([0, 1], repeat=nb_coords):
            m = [0] * 9
            for i in range(nb_coords):
                m[coords[i]] = p[i]
            l.append(tuple(m))

        return l


def launch(mdp_class, RL_tec, N=-1, precision=1e-13):
    mdp = mdp_class

    # Application of Q-Iteration to this MDP
    Qitr = RL_tec(mdp)
    Qitr.run(N=N, precision=precision)
    #
    # Compute the policy and display it
    pi = Policy(mdp, Qitr.Q)
    pi.displayPi()


# Example of execution
# (this program will not execute properly, since the class StarGathering
#  has not been defined and the class QIteration is incomplete...)
# launch(StarGathering(positive_reward=10), QIteration)

# launch(StarGathering(positive_reward=10), QLearning, N=100)

m0 = (1, 0, 0, 0, 0, 1, 0, 1, 0)
star_collector_instance = StarCollector(positive_reward=100, m0=m0)
launch(star_collector_instance, QIteration)
