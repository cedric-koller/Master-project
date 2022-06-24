import numpy as np

from utils.rules import *

def construct_transfer_matrix(rule, mu=0, beta=np.inf):
    T=np.zeros((4,4))
    if beta==np.inf:
        T=np.array([[np.exp(mu)*(rule[0]==1), np.exp(mu)*(rule[1]==1), 0, 0],
            [0, 0, rule[2]==0, rule[3]==0],
            [np.exp(mu)*(rule[4]==1), np.exp(mu)*(rule[5]==1), 0, 0],
            [0, 0, rule[6]==0, rule[7]==0]])
    else:
        T=np.array([[np.exp(mu)*np.exp(-beta*(rule[0]!=1)), np.exp(mu)*np.exp(-beta*(rule[1]!=1)), 0, 0],
            [0, 0, np.exp(-beta*(rule[2]!=0)), np.exp(-beta*(rule[3]!=0))],
            [np.exp(mu)*np.exp(-beta*(rule[4]!=1)), np.exp(mu)*np.exp(-beta*(rule[5]!=1)), 0, 0],
            [0, 0, np.exp(-beta*(rule[6]!=0)), np.exp(-beta*(rule[7]!=0))]])
    
    return T

def free_entropy_transfer_matrix(rule_array, mu=0, beta=np.inf, N=np.inf):
    eig=np.linalg.eigvals(construct_transfer_matrix(rule_array, mu, beta))
    if N==np.inf:
        return np.log(max(eig.real))
    else:
        eig=eig.real
        return np.log(eig[0]**N+eig[1]**N+eig[2]**N+eig[3]**N)/N
