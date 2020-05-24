"""
foeQ
"""
import numpy as np
from soccer import *
import random as rand
import pandas as pd
from util import *
from progress.bar import IncrementalBar as Bar

class foeQ_2Q(object):
    """
    two separate Q tables for both players are learned
    linprog on each Q table to get strategy for each player
    """

    def __init__(self, game=soccer(), \
        alpha=0.9, alpha_end=0.001, alpha_decay=0.9999954, \
        gamma=0.9, \
        epsilon=0.9, epsilon_end=0.001, epsilon_decay=0.999991, \
        maxepisode=1e5,\
        solver=None):
        self.maxepisode = maxepisode
        self.game = game
        self.nA = game.nA
        self.nS = game.nS
        self.alpha = alpha
        self.alpha_end = alpha_end
        #self.alpha_decay = alpha_decay
        self.alpha_decay = (alpha_end / alpha) ** (1. / maxepisode)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        if epsilon>0: self.epsilon_decay = (epsilon_end / epsilon) ** (1. / maxepisode)
        else: self.epsilon_decay = 0
        self.solver = solver
        #self.epsilon_decay = epsilon_decay
        self.Q = np.ones((self.nS, self.nA, self.nA, 2), dtype=float) #n of players is 2
        self.V = np.ones((self.nS, 2), dtype=float) #n of players is 2
        #self.p = np.full((self.nS, self.nA, 2), 1/self.nA, dtype=float)
        self.data = []
        self.final_policy = None

        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :, 0]
        r_matrix_B = self.Q[s, :, :, 1]

        p_A = maxmin(r_matrix_A, solver=self.solver)["x"]
        #print(list(p_A))
        v_A = list(p_A)[0]
        p_A = np.array(list(p_A)[1:])
        p_A[np.where(p_A<0)] = 0
        p_A = p_A / p_A.sum()

        p_B = maxmin(r_matrix_B.T, solver=self.solver)["x"]
        #print(list(p_B))
        v_B = list(p_B)[0]
        p_B = np.array(list(p_B)[1:])
        p_B[np.where(p_B<0)] = 0
        p_B = p_B / p_B.sum()

        #p = np.vstack((p_A, p_B)).T

        #self.p[s, :, 0] = p_A
        #self.p[s, :, 1] = p_B
        
        
        #assert v_A == v_B, "v_A =/= v_B"

        return p_A, p_B, v_A, v_B


    def compute_expected_value(self, state, v_A, v_B):
        #print(policy.reshape(-1,1))
        self.V[state, 0] = v_A
        self.V[state, 1] = v_B
        
        return np.array([v_A, v_B])

    def choose_action(self, p_A, p_B):
        #epsilon greedy
        rd = rand.random()
        if rd < self.epsilon:
            action = rand.randint(0, self.game.nJointA-1)
            #a_A, a_B = self.game.decode_action(action)
        else:
            a_A = categorical_sample(p_A)
            a_B = categorical_sample(p_B)
            action = self.game.encode_action(a_A, a_B)
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, v_A, v_B):
        a_A, a_B = self.game.decode_action(a)
        if done:
            self.Q[s, a_A, a_B, :] =\
                (1-self.alpha)*self.Q[s, a_A, a_B, :] + self.alpha * (1 - self.gamma) * np.array([r_A, r_B])
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B, :] =\
                (1-self.alpha)*self.Q[s, a_A, a_B, :] + self.alpha * ((1 - self.gamma) * np.array([r_A, r_B]) + self.gamma*self.compute_expected_value(s_prime, v_A, v_B))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("foeQ_2Q", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B, v_A, v_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B, v_A, v_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, v_A, v_B)
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
            Q_value_prime = self.get_Q_value()
            if s == self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1) and a == self.game.encode_action(a_A=2, a_B=0):
                self.data.append([T+1, Q_value_prime])
            #print("step: {}, Q: {}".format(T, Q_value_prime))
            err_Q = np.abs(Q_value_prime - Q_value)
            Q_value = Q_value_prime
            #print("step: {}, Err_Q: {}".format(T, err_Q))
            #s = s_prime
            T += 1
            if done:
                #print("yes")
                #self.game.render()
                s = self.game.reset()
                p_A, p_B, v_A, v_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_foeQ.npy", self.Q)
        p_A, p_B, _, _ = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.vstack((p_A, p_B))
        print(p_A)
        print(p_B)
        print(p_A.sum())
        print(p_B.sum())
        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0, 0]
        return Q_value

class foeQ_1Q_2LP(object):
    """
    one Q tables for A is learned
    Q table for be is assumed to be the negation of Q table for A
    linprog on Q and -Q.T to get strategy for each player
    """

    def __init__(self, game=soccer(), \
        alpha=0.9, alpha_end=0.001, alpha_decay=0.9999954, \
        gamma=0.9, \
        epsilon=0.9, epsilon_end=0.001, epsilon_decay=0.999991, \
        maxepisode=1e5,\
        solver=None):
        self.maxepisode = maxepisode
        self.game = game
        self.nA = game.nA
        self.nS = game.nS
        self.alpha = alpha
        self.alpha_end = alpha_end
        #self.alpha_decay = alpha_decay
        self.alpha_decay = (alpha_end / alpha) ** (1. / maxepisode)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        if epsilon>0: self.epsilon_decay = (epsilon_end / epsilon) ** (1. / maxepisode)
        else: self.epsilon_decay = 0
        self.solver = solver
        #self.epsilon_decay = epsilon_decay
        self.Q = np.ones((self.nS, self.nA, self.nA), dtype=float) #n of players is 2
        #self.V = np.ones((self.nS), dtype=float) #n of players is 2
        #self.p = np.full((self.nS, self.nA, 2), 1/self.nA, dtype=float)
        self.data = []

        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :]
        r_matrix_B = - r_matrix_A

        p_A = maxmin(r_matrix_A, solver=self.solver)["x"]
        #print(list(p_A))
        v_A = list(p_A)[0]
        p_A = np.array(list(p_A)[1:])
        p_A[np.where(p_A<0)] = 0
        p_A = p_A / p_A.sum()

        p_B = maxmin(r_matrix_B.T, solver=self.solver)["x"]
        #print(list(p_B))
        v_B = list(p_B)[0]
        p_B = np.array(list(p_B)[1:])
        p_B[np.where(p_B<0)] = 0
        p_B = p_B / p_B.sum()

        #p = np.vstack((p_A, p_B)).T

        #self.p[s, :, 0] = p_A
        #self.p[s, :, 1] = p_B
        
        
        #assert v_A == v_B, "v_A =/= v_B"

        return p_A, p_B, v_A, v_B

    
    def compute_expected_value(self, state, v_A, v_B):
        #print(policy.reshape(-1,1))
        #self.V[state, 0] = v_A
        #self.V[state, 1] = v_B
        
        return np.array([v_A, v_B])

    def choose_action(self, p_A, p_B):
        #epsilon greedy
        rd = rand.random()
        if rd < self.epsilon:
            action = rand.randint(0, self.game.nJointA-1)
            #a_A, a_B = self.game.decode_action(action)
        else:
            a_A = categorical_sample(p_A)
            a_B = categorical_sample(p_B)
            action = self.game.encode_action(a_A, a_B)
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, v_A, v_B):
        a_A, a_B = self.game.decode_action(a)
        if done:
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * (1 - self.gamma) * r_A
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * ((1 - self.gamma) * r_A + self.gamma * v_A)
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("foeQ_1Q_2LP", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B, v_A, v_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B, v_A, v_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, v_A, v_B)
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
            Q_value_prime = self.get_Q_value()
            if s == self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1) and a == self.game.encode_action(a_A=2, a_B=0):
                self.data.append([T+1, Q_value_prime])
            #print("step: {}, Q: {}".format(T, Q_value_prime))
            err_Q = np.abs(Q_value_prime - Q_value)
            Q_value = Q_value_prime
            #print("step: {}, Err_Q: {}".format(T, err_Q))
            s = s_prime
            T += 1
            if done:
                #print("yes")
                #self.game.render()
                s = self.game.reset()
                p_A, p_B, v_A, v_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_foeQ.npy", self.Q)
        p_A, p_B, _, _ = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.vstack((p_A, p_B))
        print(p_A)
        print(p_B)
        print(p_A.sum())
        print(p_B.sum())
        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0]
        return Q_value


class foeQ_1Q_1LP(object):
    """
    one Q tables for A is learned
    linprog on Q to get strategy for A, strategy for B is assumed to be symmetric to that for A
    """

    def __init__(self, game=soccer(), \
        alpha=0.9, alpha_end=0.001, alpha_decay=0.9999954, \
        gamma=0.9, \
        epsilon=0.9, epsilon_end=0.001, epsilon_decay=0.999991, \
        maxepisode=1e5,\
        solver=None):
        self.maxepisode = maxepisode
        self.game = game
        self.nA = game.nA
        self.nS = game.nS
        self.alpha = alpha
        self.alpha_end = alpha_end
        #self.alpha_decay = alpha_decay
        self.alpha_decay = (alpha_end / alpha) ** (1. / maxepisode)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        if epsilon>0: self.epsilon_decay = (epsilon_end / epsilon) ** (1. / maxepisode)
        else: self.epsilon_decay = 0
        self.solver = solver
        #self.epsilon_decay = epsilon_decay
        self.Q = np.ones((self.nS, self.nA, self.nA), dtype=float) #n of players is 2
        #self.V = np.ones((self.nS), dtype=float) #n of players is 2
        #self.p = np.full((self.nS, self.nA, 2), 1/self.nA, dtype=float)
        self.data = []

        pass



    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :]
        #r_matrix_B = - r_matrix_A

        p_A = maxmin(r_matrix_A, solver=self.solver)["x"]
        #print(list(p_A))
        v_A = list(p_A)[0]
        p_A = np.array(list(p_A)[1:])
        p_A[np.where(p_A<0)] = 0
        p_A = p_A / p_A.sum()

        #p_B = maxmin(r_matrix_B.T, solver=self.solver)["x"]
        #print(list(p_B))
        #v_B = list(p_B)[0]
        #p_B = np.array(list(p_B)[1:])
        #p_B[np.where(p_B<0)] = 0
        #p_B = p_B / p_B.sum()

        #p = np.vstack((p_A, p_B)).T

        #self.p[s, :, 0] = p_A
        #self.p[s, :, 1] = p_B
        
        
        #assert v_A == v_B, "v_A =/= v_B"

        return p_A, p_A, v_A, v_A


    def compute_expected_value(self, state, v_A, v_B):
        #print(policy.reshape(-1,1))
        #self.V[state, 0] = v_A
        #self.V[state, 1] = v_B
        
        return np.array([v_A, v_B])

    def choose_action(self, p_A, p_B):
        #epsilon greedy
        rd = rand.random()
        if rd < self.epsilon:
            action = rand.randint(0, self.game.nJointA-1)
            #a_A, a_B = self.game.decode_action(action)
        else:
            a_A = categorical_sample(p_A)
            a_B = categorical_sample(p_B)
            action = self.game.encode_action(a_A, a_B)
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, v_A, v_B):
        a_A, a_B = self.game.decode_action(a)
        if done:
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * (1 - self.gamma) * r_A
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * ((1 - self.gamma) * r_A + self.gamma * v_A)
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("foeQ_1Q_1LP", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B, v_A, v_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B, v_A, v_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, v_A, v_B)
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
            Q_value_prime = self.get_Q_value()
            if s == self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1) and a == self.game.encode_action(a_A=2, a_B=0):
                self.data.append([T+1, Q_value_prime])
            #print("step: {}, Q: {}".format(T, Q_value_prime))
            err_Q = np.abs(Q_value_prime - Q_value)
            Q_value = Q_value_prime
            #print("step: {}, Err_Q: {}".format(T, err_Q))
            #s = s_prime
            T += 1
            if done:
                #print("yes")
                #self.game.render()
                s = self.game.reset()
                p_A, p_B, v_A, v_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_foeQ.npy", self.Q)
        p_A, p_B, _, _ = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.vstack((p_A, p_B))
        print(p_A)
        print(p_B)
        print(p_A.sum())
        print(p_B.sum())
        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0]
        return Q_value


if __name__ == '__main__':

    print("foe-Q learner")
    print("-----------------")
    a = foeQ(epsilon=0., epsilon_end=0., maxepisode=2e5)
    a.train()
    save_results(a.data)
    #action = a.choose_action(73)
    #print(action)