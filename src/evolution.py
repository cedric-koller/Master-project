import numpy as np
import networkx as nx

def step(G, rule):
    N=G.number_of_nodes()
    G_=G.copy()
    new_state=np.zeros(N)
    for i in range(N):
        neighbours=[n for n in G.neighbors(i)]
        rule_index=2**(len(neighbours))*G_.nodes[i]['state']
        for j in range(len(neighbours)):
            rule_index+=2**j*G_.nodes[neighbours[j]]['state']
        G.nodes[i]['state']=rule[rule_index]
        
def step_game_of_life(state):
    new_state=state.copy()
    N=state.shape[0]
    for i in range(N):
        for j in range(N):
            if state[i,j]==0:
                if np.sum(state[i-1:i+2, j-1:j+2])==3:
                    new_state[i,j]=1
            else:
                if np.sum(state[i-1:i+2, j-1:j+2])-1!=2 and np.sum(state[i-1:i+2, j-1:j+2])-1!=3:
                    new_state[i,j]=0
    return new_state
        
def step_ECA(state, rule): # step with periodic boundary condition
    state_size=state.size
    new_state=np.zeros(state_size, dtype=np.int8)
    for i in range(state_size):
        left=state[(i-1)%state_size]
        middle=state[i]
        right=state[(i+1)%state_size]
        if left==1 and middle==1 and right==1:
            new_state[i]=rule[0]
        elif left==1 and middle==1 and right==0:
            new_state[i]=rule[1]
        elif left==1 and middle==0 and right==1:
            new_state[i]=rule[2]
        elif left==1 and middle==0 and right==0:
            new_state[i]=rule[3]
        elif left==0 and middle==1 and right==1:
            new_state[i]=rule[4]
        elif left==0 and middle==1 and right==0:
            new_state[i]=rule[5]   
        elif left==0 and middle==0 and right==1:
            new_state[i]=rule[6]
        elif left==0 and middle==0 and right==0:
            new_state[i]=rule[7]
    return new_state

def dynamics_ECA(initial_state, rule, n_steps=20):
    states=np.zeros((n_steps+1, initial_state.size))
    states[0,:]=initial_state
    for i in range(n_steps):
        states[i+1,:]=step_ECA(states[i,:], rule)
    return states