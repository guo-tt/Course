"""
helper function for
saving results to csv and plotting, and
linear programming to calculate minimax and CE
"""
import os
import sys
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
font = {'size': 15}
rc('font', **font)
solvers.options['show_progress'] = False
solvers.options['glpk'] = {'tm_lim': 1000} # max timeout for glpk
solvers.options['show_progress'] = False # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    prob_n = prob_n / prob_n.sum()
    #csprob_n = np.cumsum(prob_n)
    return np.random.choice(range(prob_n.shape[0]), p=prob_n)

def re_plot(data_file):
    output = pd.read_csv(os.path.join("results", "{}.csv".format(data_file)), header=0, index_col=0)
    ax1 = output[["Q-value Difference"]].plot(legend=False)
    ax1.set_xlabel("Simulation Iteration")
    ax1.set_ylabel("Q-value Difference")
    ax1.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "{}_cut.png".format(data_file)))
    plt.close()

    ax2 = output[["Q-value Difference"]].plot(legend=False)
    ax2.set_xlabel("Simulation Iteration")
    ax2.set_ylabel("Q-value Difference")
    #ax2.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "{}_uncut.png".format(data_file)))
    plt.close()

def log(learner_name, alpha, alpha_end, epsilon, epsilon_end, maxepisode, solver):
    logfile_name = os.path.join("results", "log.csv")
    if os.path.isfile(logfile_name):
        log = pd.read_csv(logfile_name, index_col=0, header=0)
        trial_num = max(list(log.index)) + 1
    else:
        trial_num = 0
        log = pd.DataFrame(columns=["learner_name", "alpha", "alpha_end", "epsilon", "epsilon_end", "maxepisode", "solver"])
    #entry = {"learner_name":[learner_name], "alpha":[alpha], "alpha_end":[alpha_end], "epsilon":[epsilon], "epsilon_end":[epsilon_end], "maxepisode":[maxepisode], "solver":[solver]}
    entry = [learner_name, alpha, alpha_end, epsilon, epsilon_end, maxepisode, solver]
    #entry = pd.DataFrame(entry)
    #print(list(entry.index))
    log.loc[trial_num] = entry
    #print(log)
    log.to_csv(logfile_name)
    return trial_num


def save_results(data, final_policy, Qtable, trial_num):
    output = pd.DataFrame(data=np.array(data), columns=["Simulation Iteration", "Q-Value"])
    output.set_index("Simulation Iteration", inplace=True)
    output["Q-value Difference"] = np.abs(output.diff(periods=1, axis=0))

    output.to_csv(os.path.join("results", "Qdifference_{}.csv".format(trial_num)))
    ax = output[["Q-value Difference"]].plot(legend=False)
    ax.set_xlabel("Simulation Iteration")
    ax.set_ylabel("Q-value Difference")
    ax.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "Qdifference_{}_cut.png".format(trial_num)))
    plt.close()

    ax2 = output[["Q-value Difference"]].plot(legend=False)
    ax2.set_xlabel("Simulation Iteration")
    ax2.set_ylabel("Q-value Difference")
    #ax2.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "Qdifference_{}_uncut.png".format(trial_num)))
    plt.close()

    df_final_policy = pd.DataFrame(final_policy)
    df_final_policy.to_csv(os.path.join("results", "final_policy_{}.txt".format(trial_num)), header=False, index=False)

    np.save(os.path.join("results", "Qtable_{}.npy".format(trial_num)), Qtable)


def maxmin(A, solver=None):
    nA = A.shape[0] #number of actions for one player
    # minimize matrix c: minimize c*x
    c = np.array([-1] + [0] * nA, dtype=float)
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(A, dtype=float).T # reformat each variable is in a row
    G *= -1 # minimization constraint
    G = np.vstack([G, np.eye(nA) * -1]) # > 0 constraint for all vars
    utility = np.hstack((np.ones(nA, dtype=float), np.zeros(nA, dtype=float))) # utility, 1 for rationality constraints, 0 for positive probability constraints
    G = np.hstack((utility.reshape(-1,1), G)) # insert utility column
    G = matrix(G)
    h = np.zeros(nA * 2, dtype=float)
    h = matrix(h)
    # contraints Ax = b: sum of all probabilites is 1
    A = [0] + [1] * nA
    A = np.matrix(A, dtype=float)
    A = matrix(A)
    b = np.matrix(1, dtype=float)
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

def ce(A, solver=None): #correlated equilibrium
    nA = A.shape[0] #number of joint actions for two players
    # maximize matrix c
    c = A.sum(axis=1) # sum of payoffs for both players
    c = matrix(c)
    c *= -1 # cvxopt minimizes so *-1 to maximize the sum of both players' reward
    # constraints G*x <= h
    G = create_G_matrix_CE(A=A)
    G = np.vstack((G, -1 * np.eye(nA))) # > 0 constraint for all vars
    h = np.zeros(G.shape[0], dtype=float)
    G = matrix(G)
    h = matrix(h)
    # contraints Ax = b
    A = np.matrix([1] * nA, dtype=float)
    A = matrix(A)
    b = np.matrix(1, dtype=float)
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

def create_G_matrix_CE(A): #rationality constraints
    nA = int(len(A) ** 0.5) #number of actions for one player
    G = []
    # row player
    for i in range(nA): # action row i
        for j in range(nA): # action row j
            if i != j:
                temp = [0] * len(A)
                for k in range(nA):
                    temp[i * nA + k] = (
                        - A[i * nA + k][0]
                        + A[j * nA + k][0])
                G += [temp]
    # col player
    for i in range(nA): # action column i
        for j in range(nA): # action column j
            if i != j:
                temp = [0] * len(A)
                for k in range(nA):
                    temp[i + (k * nA)] = (
                        - A[i + (k * nA)][1] 
                        + A[j + (k * nA)][1])
                G += [temp]
    return np.matrix(G, dtype=float)


if __name__ == '__main__':

    print("this is the code for the saving results and linear programming for maxmin and ce")

    #data_file = sys.argv[1]
    #re_plot(data_file)




