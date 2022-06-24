import pandas as pd

from src.BP import *
from src.transfer_matrix import *
from utils.rules import *


def game_of_life_rs_calculation(d_list=[8], mu_list=[0], tol=1e-12, max_iter=10000, damping_parameter=0.5, verbose=0, init_psi=None):
    d_list_=[]
    mu_list_=[]
    psi_list=[]
    marginal_list=[]
    phi_list=[]
    rho_list=[]
    s_list=[]
    
    with tqdm(total=len(d_list)*len(mu_list)) as pbar:
            for d in d_list:
                rule=game_of_life_rule(d=d)
                for mu in mu_list:
                    d_list_.append(d)
                    mu_list_.append(mu)
                    psi=BP(rule=rule, d=d, mu=mu, tol=tol, max_iter=max_iter, damping_parameter=damping_parameter, verbose=verbose, init_psi=init_psi)
                    psi_list.append(psi)
                    marginal_list.append(marginals(psi))
                    phi=compute_phi(psi, d, rule, mu)
                    phi_list.append(phi)
                    rho=density(psi, d, rule, mu)
                    rho_list.append(rho)
                    s=entropy(phi, rho, mu)
                    s_list.append(s)
                    pbar.update(1)

    dic= {'d': d_list_, 'mu': mu_list_, 'messages': psi_list, 'marginals': marginal_list, 'free_entropy': phi_list, 'density': rho_list, 'entropy': s_list }
    return pd.DataFrame.from_dict(dic)

def rs_calculation(rule_list, d_list=[8], mu_list=[0], tol=1e-12, max_iter=10000, damping_parameter=0.5, verbose=0, init_psi=None):
    rule_list_=[]
    d_list_=[]
    mu_list_=[]
    marginal_list=[]
    psi_list=[]
    phi_list=[]
    rho_list=[]
    s_list=[]
    
    with tqdm(total=len(rule_list)*len(d_list)*len(mu_list)) as pbar:
        for rule in rule_list:
            for d in d_list:
                for mu in mu_list:
                    d_list_.append(d)
                    mu_list_.append(mu)
                    rule_list_.append(new_notation_to_wolfram_notation(rule))
                    psi=BP(rule=rule, d=d, mu=mu, tol=tol, max_iter=max_iter, damping_parameter=damping_parameter, verbose=verbose, init_psi=init_psi)
                    psi_list.append(psi)
                    marginal_list.append(marginals(psi))
                    phi=compute_phi(psi, d, rule, mu)
                    phi_list.append(phi)
                    rho=density(psi, d, rule, mu)
                    rho_list.append(rho)
                    s=entropy(phi, rho, mu)
                    s_list.append(s)
                    pbar.update(1)

    dic= {'rule': rule_list_, 'd': d_list_, 'mu': mu_list_, 'messages': psi_list, 'marginals': marginal_list, 'free_entropy': phi_list, 'density': rho_list, 'entropy': s_list }
    return pd.DataFrame.from_dict(dic)

def transfer_matrix_density_entropy(rule_array, mu_list, beta=np.inf): # rule_array: binary notation
    free_entropy_list=[]
    for mu in mu_list:
        free_entropy_list.append(free_entropy_transfer_matrix(rule_array, mu, beta))
    density_list=density_numerical(np.array(free_entropy_list), np.array(mu_list))
    entropy_list=entropy_from_arrays(np.array(free_entropy_list), np.array(density_list), np.array(mu_list))
    return density_list, entropy_list, free_entropy_list

