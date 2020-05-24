"""
ceQ
"""
import numpy as np
from soccer import *
import random as rand
import pandas as pd
from util import *
from progress.bar import IncrementalBar as Bar


class ceQ(object):
    """
    two separate Q tables for both players are learned
    """

    def __init__(self, game=soccer(), \
        alpha=0.9, alpha_end=0.001, alpha_decay=0.9999954, \
        gamma=0.9, \
        epsilon=0.9, epsilon_end=0.001, epsilon_decay=0.999991, \
        maxepisode=1e5,\
        solver=None):
        self.maxepisode = maxepisode
        self.game = game
        self.nA = game.nJointA
        self.nS = game.nS
        self.alpha = alpha
        self.alpha_end = alpha_end
        #self.alpha_decay = alpha_decay
        self.alpha_decay = (alpha_end / alpha) ** (1. / maxepisode)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.solver = solver
        if epsilon>0: self.epsilon_decay = (epsilon_end / epsilon) ** (1. / maxepisode)
        else: self.epsilon_decay = 0
        #self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((self.nS, self.nA, 2), dtype=float) #n of players is 2
        self.V = np.zeros((self.nS, 2), dtype=float) #n of players is 2
        #self.p = np.full((self.nS, self.nA), 1/self.nA, dtype=float)
        self.data = []


    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix = self.Q[s, :, :]
        p = ce(r_matrix, solver=self.solver)["x"]
        p = np.array(list(p))
        p[np.where(p<0)] = 0
        p = p / p.sum()
        #self.p[s, :] = p
        return p

    def compute_expected_value(self, s, policy):
        #print(policy.reshape(-1,1))
        v = (self.Q[s, :, :] * policy.reshape(-1,1)).sum(axis=0)
        self.V[s] = v
        #print(v)
        #print(v)
        return v

    def choose_action(self, policy):
        rd = rand.random()
        if rd < self.epsilon:
            action = rand.randint(0, self.game.nJointA-1)
            #a_A, a_B = self.game.decode_action(action)
        else:
            action = categorical_sample(policy)
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, p):
        if done:
            self.Q[s, a, :] =\
                (1-self.alpha)*self.Q[s, a, :] + self.alpha * (1 - self.gamma) * np.array([r_A, r_B])
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a, :] =\
                (1-self.alpha)*self.Q[s, a, :] + self.alpha * ((1 - self.gamma) * np.array([r_A, r_B]) + self.gamma*self.compute_expected_value(s_prime, p))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("ceQ", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:
            a = self.choose_action(p)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, p)
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
            Q_value_prime = self.get_Q_value()
            if s == self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1) and a == self.game.encode_action(a_A=2, a_B=0):
                self.data.append([T+1, Q_value_prime])
            #print("step: {}, Q: {}".format(T, Q_value_prime))
            err_Q = np.abs(Q_value_prime - Q_value)
            Q_value = Q_value_prime
            #print("step: {}, Err_Q: {}".format(T, err_Q))
            
            T += 1
            if done:
                #print("yes")
                #self.game.render()
                s = self.game.reset()
                p = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_ceQ.npy", self.Q)
        final_policy = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        p_A = final_policy.reshape(5,5).sum(axis=1)
        p_B = final_policy.reshape(5,5).sum(axis=0)
        self.final_policy = np.vstack((p_A, p_B))
        print(p_A)
        print(p_B)
        print(final_policy.sum())
        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), self.game.encode_action(a_A=2, a_B=0), 0]
        return Q_value




if __name__ == '__main__':

    print("ce-Q learner")
    print("-----------------")
    a = ceQ(epsilon=0., epsilon_end=0., maxepisode=2e5)
    a.train()
    save_results(a.data)
    #action = a.choose_action(73)
    #print(action)