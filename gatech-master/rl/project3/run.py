from ceQ import *
from friendQ import *
from foeQ import *
from Q import *
from util import *
import sys
solvers.options['show_progress'] = False
solvers.options['glpk'] = {'tm_lim': 1000} # max timeout for glpk
solvers.options['show_progress'] = False # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

def run_learner(learner_name="friendQ", alpha=0.9, alpha_end=0.001, epsilon=1., epsilon_end=1., maxepisode=2e5, solver=None):
    trial_num = log(learner_name=learner_name, alpha=alpha, alpha_end=alpha_end, epsilon=epsilon, epsilon_end=epsilon_end, maxepisode=maxepisode, solver=solver)
    learner = eval(learner_name)
    a = learner(alpha=alpha, alpha_end=alpha_end, epsilon=epsilon, epsilon_end=epsilon_end, maxepisode=maxepisode, solver=solver)
    a.train()
    save_results(a.data, a.final_policy, a.Q, trial_num=trial_num)





if __name__ == '__main__':
    name = str(sys.argv[1])
    alpha = float(sys.argv[2])
    alpha_end = float(sys.argv[3])
    epsilon = float(sys.argv[4])
    epsilon_end = float(sys.argv[5])
    maxepisode = float(sys.argv[6])

    #run_learner("ceQ", alpha=0.1, epsilon=1., epsilon_end=0.001, maxepisode=10e5)
    #run_learner("friendQ_2Q", alpha=0.1, epsilon=1., epsilon_end=0.001, maxepisode=10e5)
    #run_learner("friendQ_1Q", alpha=0.1, epsilon=1., epsilon_end=0.001, maxepisode=10e5)
    #run_learner("foeQ_1Q", alpha=1.0, epsilon=1., epsilon_end=0.001, maxepisode=10e5, solver=None)
    run_learner(learner_name=name, alpha=alpha, alpha_end=alpha_end, epsilon=epsilon, epsilon_end=epsilon_end, maxepisode=maxepisode, solver=None)
    
    #action = a.choose_action(73)
    #print(action)