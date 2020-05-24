"""
friendQ
"""
import numpy as np
from soccer import *
import random as rand
import pandas as pd
from util import *
from progress.bar import IncrementalBar as Bar

class friendQ_2Q(object):
    """
    two separate Q tables for both players are learned
    linprog on each Q table to get strategy for each player
    """

    def __init__(self, game=soccer(),\
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
        self.Q = np.ones((self.nS, self.nA, self.nA, 2), dtype=float) #n of players is 2
        self.V = np.ones((self.nS, 2), dtype=float) #n of players is 2
        self.data = []
        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :, 0]
        r_matrix_B = self.Q[s, :, :, 1]

        p_A = np.zeros(self.nA, dtype=float)
        winner_A = np.where(r_matrix_A == r_matrix_A.max())[0]
        #winner_A = np.unravel_index(r_matrix_A.argmax(), r_matrix_A.shape)[0]
        p_A[winner_A] = 1
        p_A = p_A / p_A.sum()

        p_B = np.zeros(self.nA, dtype=float)
        winner_B = np.where(r_matrix_B == r_matrix_B.max())[1]
        #winner_B = np.unravel_index(r_matrix_B.argmax(), r_matrix_B.shape)[1]
        p_B[winner_B] = 1
        p_B = p_B / p_B.sum()
        #print(p_A, p_B)
        return p_A, p_B

    def compute_expected_value(self, s, p_A, p_B):
        p = np.outer(p_A, p_B)
        #print(policy.reshape(-1,1))
        v_A = (self.Q[s, :, :, 0] * p).sum()
        v_B = (self.Q[s, :, :, 1] * p).sum()
        self.V[s] = np.array([v_A, v_B])
        #print(v)
        #print(v)
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
        #print(self.game.decode_action(action))
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, p_A, p_B):
        a_A, a_B = self.game.decode_action(a)
        #r_A, r_B = np.abs(r_A), np.abs(r_B) #this is wrong
        if done:
            self.Q[s, a_A, a_B, :] =\
                (1-self.alpha)*self.Q[s, a_A, a_B, :] + self.alpha * (1 - self.gamma) * np.array([r_A, r_B])
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B, :] =\
                (1-self.alpha)*self.Q[s, a_A, a_B, :] + self.alpha * ((1 - self.gamma) * np.array([r_A, r_B]) + self.gamma*self.compute_expected_value(s_prime, p_A, p_B))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("friendQ_2Q", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, p_A, p_B)
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
                p_A, p_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_friendQ_2Q.npy", self.Q)
        final_policy = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.array(final_policy)
        print(final_policy[0])
        print(final_policy[1])
        print(final_policy[0].sum())
        print(final_policy[1].sum())
        #print(final_policy.reshape(5,5).sum(axis=1))
        #print(final_policy.reshape(5,5).sum(axis=0))
        #print(final_policy.sum())

        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        #Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), self.game.encode_action(a_A=2, a_B=0), 0]
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0, 0]
        return Q_value

class friendQ_1Q(object):
    """
    one Q tables for A is learned
    Q table for be is assumed to be the negation of Q table for A
    linprog on Q and -Q.T to get strategy for each player
    """

    def __init__(self, game=soccer(),\
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
        self.Q = np.ones((self.nS, self.nA, self.nA), dtype=float) #n of players is 2
        self.V = np.ones((self.nS), dtype=float) #n of players is 2
        self.data = []
        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :]
        r_matrix_B = - self.Q[s, :, :]

        p_A = np.zeros(self.nA, dtype=float)
        winner_A = np.where(r_matrix_A == r_matrix_A.max())[0]
        
        #winner_A = np.unravel_index(r_matrix_A.argmax(), r_matrix_A.shape)[0]
        p_A[winner_A] = 1
        p_A = p_A / p_A.sum()

        
        p_B = np.zeros(self.nA, dtype=float)
        winner_B = np.where(r_matrix_B == r_matrix_B.max())[1]
        #winner_B = np.unravel_index(r_matrix_B.argmax(), r_matrix_B.shape)[1]
        p_B[winner_B] = 1
        p_B = p_B / p_B.sum()
        #print(p_A, p_B)
        return p_A, p_B

    def compute_expected_value(self, s, p_A, p_B):
        p = np.outer(p_A, p_B)
        #print(policy.reshape(-1,1))
        v_A = (self.Q[s, :, :] * p).sum()
        self.V[s] = v_A
        #print(v)
        #print(v)
        return v_A

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
        #print(self.game.decode_action(action))
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, p_A, p_B):
        a_A, a_B = self.game.decode_action(a)
        #r_A, r_B = np.abs(r_A), np.abs(r_B) #this is wrong
        if done:
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * (1 - self.gamma) * r_A
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * ((1 - self.gamma) * r_A + self.gamma*self.compute_expected_value(s_prime, p_A, p_B))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("friendQ_1Q", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, p_A, p_B)
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
                p_A, p_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_friendQ_1Q.npy", self.Q)
        final_policy = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.array(final_policy)
        print(final_policy[0])
        print(final_policy[1])
        print(final_policy[0].sum())
        print(final_policy[1].sum())
        #print(final_policy.reshape(5,5).sum(axis=1))
        #print(final_policy.reshape(5,5).sum(axis=0))
        #print(final_policy.sum())

        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        #Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), self.game.encode_action(a_A=2, a_B=0), 0]
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0]
        return Q_value


class friendQ_2Q_alt(object):
    """
    two separate Q tables for both players are learned
    linprog on each Q table to get strategy for each player
    alt means: here V(s') is evaluated by max, instead of expected value
    """

    def __init__(self, game=soccer(),\
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
        self.Q = np.ones((self.nS, self.nA, self.nA, 2), dtype=float) #n of players is 2
        self.V = np.ones((self.nS, 2), dtype=float) #n of players is 2
        self.data = []
        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :, 0]
        r_matrix_B = self.Q[s, :, :, 1]

        p_A = np.zeros(self.nA, dtype=float)
        #winner_A = np.where(r_matrix_A == r_matrix_A.max())[0]
        winner_A = np.unravel_index(r_matrix_A.argmax(), r_matrix_A.shape)[0]
        p_A[winner_A] = 1
        p_A = p_A / p_A.sum()

        p_B = np.zeros(self.nA, dtype=float)
        #winner_B = np.where(r_matrix_B == r_matrix_B.max())[1]
        winner_B = np.unravel_index(r_matrix_B.argmax(), r_matrix_B.shape)[1]
        p_B[winner_B] = 1
        p_B = p_B / p_B.sum()
        #print(p_A, p_B)
        return p_A, p_B

    def compute_expected_value0(self, s, p_A, p_B):
        p = np.outer(p_A, p_B)
        #print(policy.reshape(-1,1))
        v_A = (self.Q[s, :, :, 0] * p).sum()
        v_B = (self.Q[s, :, :, 1] * p).sum()
        self.V[s] = np.array([v_A, v_B])
        #print(v)
        #print(v)
        return np.array([v_A, v_B])

    def compute_expected_value(self, s, p_A, p_B):
        #p = np.outer(p_A, p_B)
        #print(policy.reshape(-1,1))
        #v_A = (self.Q[s, :, :, 0] * p).sum()
        #v_B = (self.Q[s, :, :, 1] * p).sum()
        v_A = self.Q[s, :, :, 0].max()
        v_B = self.Q[s, :, :, 1].max()
        self.V[s] = np.array([v_A, v_B])
        #print(v)
        #print(v)
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
        #print(self.game.decode_action(action))
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, p_A, p_B):
        a_A, a_B = self.game.decode_action(a)
        #r_A, r_B = np.abs(r_A), np.abs(r_B) #this is wrong
        if done:
            self.Q[s, a_A, a_B, :] =\
                (1-self.alpha)*self.Q[s, a_A, a_B, :] + self.alpha * (1 - self.gamma) * np.array([r_A, r_B])
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B, :] =\
                (1-self.alpha)*self.Q[s, a_A, a_B, :] + self.alpha * ((1 - self.gamma) * np.array([r_A, r_B]) + self.gamma*self.compute_expected_value(s_prime, p_A, p_B))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("friendQ_2Q", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, p_A, p_B)
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
                p_A, p_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_friendQ_2Q.npy", self.Q)
        final_policy = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.array(final_policy)
        print(final_policy[0])
        print(final_policy[1])
        print(final_policy[0].sum())
        print(final_policy[1].sum())
        #print(final_policy.reshape(5,5).sum(axis=1))
        #print(final_policy.reshape(5,5).sum(axis=0))
        #print(final_policy.sum())

        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        #Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), self.game.encode_action(a_A=2, a_B=0), 0]
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0, 0]
        return Q_value

class friendQ_1Q_alt(object):
    """
    one Q tables for A is learned
    Q table for be is assumed to be the negation of Q table for A
    linprog on Q and -Q.T to get strategy for each player
    alt means: here V(s') is evaluated by max, instead of expected value
    """

    def __init__(self, game=soccer(),\
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
        self.Q = np.ones((self.nS, self.nA, self.nA), dtype=float) #n of players is 2
        self.V = np.ones((self.nS), dtype=float) #n of players is 2
        self.data = []
        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix_A = self.Q[s, :, :]
        r_matrix_B = - self.Q[s, :, :]

        p_A = np.zeros(self.nA, dtype=float)
        #winner_A = np.where(r_matrix_A == r_matrix_A.max())[0]
        
        winner_A = np.unravel_index(r_matrix_A.argmax(), r_matrix_A.shape)[0]
        p_A[winner_A] = 1
        p_A = p_A / p_A.sum()

        
        p_B = np.zeros(self.nA, dtype=float)
        #winner_B = np.where(r_matrix_B == r_matrix_B.max())[1]
        winner_B = np.unravel_index(r_matrix_B.argmax(), r_matrix_B.shape)[1]
        p_B[winner_B] = 1
        p_B = p_B / p_B.sum()
        #print(p_A, p_B)
        return p_A, p_B

    def compute_expected_value0(self, s, p_A, p_B):
        p = np.outer(p_A, p_B)
        #print(policy.reshape(-1,1))
        v_A = (self.Q[s, :, :] * p).sum()
        self.V[s] = v_A
        #print(v)
        #print(v)
        return v_A

    def compute_expected_value(self, s, p_A, p_B):
        #p = np.outer(p_A, p_B)
        #print(policy.reshape(-1,1))
        #v_A = (self.Q[s, :, :] * p).sum()
        v_A = self.Q[s, :, :].max()
        self.V[s] = v_A
        #print(v)
        #print(v)
        return v_A

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
        #print(self.game.decode_action(action))
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done, p_A, p_B):
        a_A, a_B = self.game.decode_action(a)
        #r_A, r_B = np.abs(r_A), np.abs(r_B) #this is wrong
        if done:
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * (1 - self.gamma) * r_A
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a_A, a_B] =\
                (1-self.alpha)*self.Q[s, a_A, a_B] + self.alpha * ((1 - self.gamma) * r_A + self.gamma*self.compute_expected_value(s_prime, p_A, p_B))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("friendQ_1Q", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append([T, Q_value])
        s = self.game.reset()
        p_A, p_B = self.gen_policy(s)
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(p_A, p_B)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            p_A, p_B = self.gen_policy(s_prime)
            self.learn(s, a, s_prime, r_A, r_B, done, p_A, p_B)
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
                p_A, p_B = self.gen_policy(s)
            else:
                s = s_prime
            bar.next()
        bar.finish()

        #np.save("Qtable_friendQ_1Q.npy", self.Q)
        final_policy = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        self.final_policy = np.array(final_policy)
        print(final_policy[0])
        print(final_policy[1])
        print(final_policy[0].sum())
        print(final_policy[1].sum())
        #print(final_policy.reshape(5,5).sum(axis=1))
        #print(final_policy.reshape(5,5).sum(axis=0))
        #print(final_policy.sum())

        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        #Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), self.game.encode_action(a_A=2, a_B=0), 0]
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), 2, 0]
        return Q_value


class friendQ_joint(object):

    def __init__(self, game=soccer(),\
        alpha=0.9, alpha_end=0.001, alpha_decay=0.9999954, \
        gamma=0.9, \
        epsilon=0.9, epsilon_end=0.001, epsilon_decay=0.999991, \
        maxepisode=4e5):
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
        if epsilon>0: self.epsilon_decay = (epsilon_end / epsilon) ** (1. / maxepisode)
        else: self.epsilon_decay = 0
        #self.epsilon_decay = epsilon_decay
        self.Q = np.ones((self.nS, self.nA, 2), dtype=float) #n of players is 2
        self.V = np.ones((self.nS, 2), dtype=float) #n of players is 2
        self.data = []
        pass

    def gen_policy(self, s): 
        #generate correlated policy based on Q values of both players, 
        #this policy gives joint actions, instead of choosing individual actions separately
        #print(self.Q.shape)

        r_matrix = self.Q[s, :, 0]
        p = np.zeros(self.nA, dtype=float)
        p[np.where(r_matrix == r_matrix.max())] = 1
        p /= p.sum()
        return p

    def compute_expected_value(self, s):
        policy = self.gen_policy(s)
        #print(policy.reshape(-1,1))
        v = (self.Q[s, :, :] * policy.reshape(-1,1)).sum(axis=0)
        self.V[s] = v
        #print(v)
        #print(v)
        return v

    def choose_action(self, s):
        #epsilon greedy
        rd = rand.random()
        if rd < self.epsilon:
            action = rand.randint(0, self.game.nJointA-1)
            #a_A, a_B = self.game.decode_action(action)

        else:
            policy = self.gen_policy(s)
            action = categorical_sample(policy)

            """
            r_matrix_A = self.Q[s, :, 0]
            r_matrix_B = self.Q[s, :, 1]

            p_A = np.zeros(self.nA, dtype=float)
            winner = np.where(r_matrix_A == r_matrix_A.max())
            #p_A[np.unravel_index(r_matrix_A.argmax(), r_matrix_A.shape)[0]] = 1
            p_A[winner] = 1
            p_A = p_A / p_A.sum()
            p_A = p_A.reshape(5,5).sum(axis=1)

            p_B = np.zeros(self.nA, dtype=float)
            winner = np.where(r_matrix_B == r_matrix_B.max())
            #p_B[np.unravel_index(r_matrix_B.argmax(), r_matrix_B.shape)[1]] = 1
            p_B[winner] = 1
            p_B = p_B / p_B.sum()
            p_B = p_B.reshape(5,5).sum(axis=0)

            a_A = categorical_sample(p_A)
            a_B = categorical_sample(p_B)
            action = self.game.encode_action(a_A, a_B)
            """
        return action


    def learn(self, s, a, s_prime, r_A, r_B, done):
        if done:
            self.Q[s, a, :] =\
                (1-self.alpha)*self.Q[s, a, :] + self.alpha * (1 - self.gamma) * np.array([r_A, r_B])
            
        else:
            #print("s is {}".format(s))
            #print("a is {}".format(a))
            self.Q[s, a, :] =\
                (1-self.alpha)*self.Q[s, a, :] + self.alpha * ((1 - self.gamma) * np.array([r_A, r_B]) + self.gamma*self.compute_expected_value(s_prime))
        pass

    def train(self, ):
        T = 0
        print("start training: {}_{}_{}_{}_{}_{}".format("friendQ_joint", self.alpha, self.alpha_end, self.epsilon, self.epsilon_end, self.maxepisode))
        Q_value = self.get_Q_value()
        self.data.append(Q_value)
        s = self.game.reset()
        bar = Bar('Training', max=self.maxepisode, suffix='%(index)d/%(max)d - %(elapsed)ds/%(eta)ds')
        while T < self.maxepisode:

            a = self.choose_action(s)
            #take action:
            s_prime, r_A, r_B, done, _ = self.game.step_encoded_action(a)
            #self.game.render()
            self.learn(s, a, s_prime, r_A, r_B, done)
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
            Q_value_prime = self.get_Q_value()
            self.data.append(Q_value_prime)
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
            else:
                s = s_prime
            bar.next()
        bar.finish()


        final_policy = self.gen_policy(self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1))
        p_A = final_policy.reshape(5,5).sum(axis=1)
        p_B = final_policy.reshape(5,5).sum(axis=0)
        self.final_policy = np.vstack((p_A, p_B))
        print(p_A)
        print(p_B)
        print(final_policy)
        print(final_policy.sum())
        fp = np.where(final_policy==1)
        print(fp[0][0])
        print(self.game.decode_action(fp[0][0]))
        pass

    def get_Q_value(self, ): #get the Q value of player A at initial state, action of A move south, B stick
        Q_value = self.Q[self.game.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1), self.game.encode_action(a_A=2, a_B=0), 0]
        return Q_value


if __name__ == '__main__':

    print("friend-Q learner")
    print("-----------------")
    a = friendQ(epsilon=0., epsilon_end=0., maxepisode=2e5)
    a.train()
    save_results(a.data)
    #action = a.choose_action(73)
    #print(action)

