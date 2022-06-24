import numpy as np
import itertools
import pickle
import pandas as pd
from tqdm import tqdm
from utils.rules import *


# Note: one uses another convention than Wolfram
def respect_rule(i, j, rest_config, rule, d):
    rule_index=2**d*i+2**(d-1)*j
    for idx in range(d-1):
        rule_index+=2**(idx)*rest_config[-idx]
    return True if i==rule[rule_index] else False

def BP(rule, d, mu=0, tol=1e-12, max_iter=10000, damping_parameter=0.5, verbose=0, init_psi=None):
    psi=np.zeros((2,2))
    if init_psi is None:
        psi=np.random.uniform(size=(2,2))
        psi=psi/np.sum(psi)
    else:
        psi=init_psi
    
    permutations=np.array(list(itertools.product([0,1], repeat=d-1)))
    
    for t in range(max_iter):
        psi_new=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                for perm in permutations:
                    mult=1
                    for k in perm:
                        mult*=psi[k,i]
                    psi_new[i,j]+=np.exp(mu*i)*respect_rule(i, j, perm, rule, d)*mult
                    
        if np.sum(psi_new)!=0:
            psi_new=psi_new/np.sum(psi_new)
        else:
            psi_new=np.array([[0.25, 0.25],[0.25, 0.25]])
        
        psi_new=damping_parameter*psi+(1-damping_parameter)*psi_new
        
        Delta=np.linalg.norm(psi_new-psi)
        if verbose==2 and t%100==0:
            print("Iter : ", t+1, " Delta : ", Delta)
        psi=psi_new
        if Delta<tol:
            break
    
    if t==max_iter-1:
        print("No convergence for rule "+str(new_notation_to_wolfram_notation(rule))+" ! Final error: "+ str(Delta))
    else:
        if verbose>=1:
            print("Converged ! Number of iteration "+str(t))
    
    return psi

def compute_phi(psi, d, rule, mu=0):
    phi_=0
    phi__=0
    permutations=np.array(list(itertools.product([0,1], repeat=d-1)))
    for i in range(2):
        for j in range(2):
            for perm in permutations:
                mult=psi[j,i]
                for k in perm:
                    mult*=psi[k,i]
                phi_+=np.exp(mu*i)*respect_rule(i, j, perm, rule, d)*mult
            phi__+=psi[i,j]*psi[j,i]
    if phi_==0:
        phi_=1e-16
    if phi__==0:
        phi__=1e-16
    return np.log(phi_)-d/2*np.log(phi__)

def density_numerical(phi_list, mu_list):
    N=phi_list.size
    rho=np.zeros(N-1)
    for i in range(N-1):
        rho[i]=(phi_list[i+1]-phi_list[i])/(mu_list[i+1]-mu_list[i])
    return rho

def density(psi, d, rule, mu=0):
    # Density is the derivative of Z^i with respect to μ
    numerator=0
    denominator=0
    permutations=np.array(list(itertools.product([0,1], repeat=d-1)))
    for i in range(2):
        for j in range(2):
            for perm in permutations:
                mult=psi[j,i]
                for k in perm:
                    mult*=psi[k,i]
                numerator+=i*np.exp(mu*i)*respect_rule(i, j, perm, rule, d)*mult
                denominator+=np.exp(mu*i)*respect_rule(i, j, perm, rule, d)*mult
    if numerator==0:
        return 0
    return numerator/denominator


def density_from_arrays(psi_list, mu_list, d, rule):
    N=mu_list.size
    rho_list=np.zeros(N)
    for i in range(N):
        rho_list[i]=density(psi_list[i], d, rule, mu_list[i])
    return rho_list

def entropy(phi, rho, mu):
    return phi-mu*rho

def entropy_from_arrays(phi_list, rho_list, mu_list):
    N=rho_list.size
    entropy=np.zeros(N)
    for i in range(N):
        entropy[i]=phi_list[i]-mu_list[i]*rho_list[i]
    return entropy

def marginals(psi):
    mu=np.zeros((2,2))
    mu[0,0]=psi[0,0]**2
    mu[1,0]=psi[1,0]*psi[0,1]
    mu[0,1]=psi[0,1]*psi[1,0]
    mu[1,1]=psi[1,1]**2
    return mu/np.sum(mu) if np.sum(mu)!=0 else np.array([[0.25, 0.25],[0.25, 0.25]])



# ================================================================================================================================

import torch

class BP_OT:
    def __init__(self, rule, mu=0, tol=1e-12, max_iter=10000, damping_parameter=0.8, psi_init=None, device='cuda'):
        if device=='cuda':
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        self.rule=rule
        self.mu=mu
        self.tol=tol
        self.max_iter=max_iter
        self.damping_parameter=damping_parameter
        self.psi=psi_init
        if self.psi is None:
            self.psi=np.random.uniform(size=(2,2))
            self.psi=self.psi/np.sum(self.psi)
        self.device=device
            
        self.psi_old=self.psi.copy()
        
        self.d=len(rule)-1
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d-1)))
        
        self.phi=None
        self.rho=None
        self.s=None
        
        self.stability=None
        self.physical=None
        
        self.fixed_points=None
        self.all_phi=None
        self.all_rho=None
        self.all_s=None
        
        self.all_stabilities=None
        self.all_physical=None
        
    def __repr__(self):
        description="Instance of class \'BP_OT\'\nRule : "+str(self.rule)+"\nμ =  "+str(self.mu)
        if self.phi is not None:
            description+='\nφ = '+str(self.phi)
            description+='\nρ = '+str(self.rho)
            description+='\ns = '+str(self.s)
            if self.physical is not None:
                description+='\nPhysical: '+str(self.physical)
            if  self.stability is not None:
                description+='\nstable: '+str(self.stability)

        return description

    def respect_rule(self, i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
        
    def step(self):
        self.psi_old=self.psi.copy()
        
        psi_new=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=1
                    for k in perm:
                        mult*=self.psi[k,i]
                    psi_new[i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    
        if np.sum(psi_new)!=0:
            psi_new=psi_new/np.sum(psi_new)
        else:
            psi_new=np.array([[0.25, 0.25],[0.25, 0.25]])
        
        self.psi=self.damping_parameter*self.psi+(1-self.damping_parameter)*psi_new
        
    def diff(self):
        return np.linalg.norm(self.psi-self.psi_old)
    
    def update_observables(self):
        phi_=0
        phi__=0
        numerator=0
        denominator=0
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=self.psi[j,i]
                    for k in perm:
                        mult*=self.psi[k,i]
                    phi_+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    numerator+=i*np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    denominator+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                phi__+=self.psi[i,j]*self.psi[j,i]
        if phi_==0:
            phi_=1e-16
        if phi__==0:
            phi__=1e-16
        self.phi=np.log(phi_)-self.d/2*np.log(phi__)
        

        if numerator==0:
            self.rho=0
        else:
            self.rho=numerator/denominator
            
        self.s=self.phi-self.mu*self.rho
        
    def marginals(self):
        marginals=np.zeros((2,2))
        marginals[0,0]=self.psi[0,0]**2
        marginals[1,0]=self.psi[1,0]*self.psi[0,1]
        marginals[0,1]=self.psi[0,1]*self.psi[1,0]
        marginals[1,1]=self.psi[1,1]**2
        return marginals/np.sum(marginals) if np.sum(marginals)!=0 else np.array([[0.25, 0.25],[0.25, 0.25]])
    
    def run(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            self.step()
            if self.diff()<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
        self.update_observables()

            
    def find_all_fixed_points_not_vectorized(self, grid_discretisation=15, precision=8, num_random_search=0):
        self.fixed_points=[]
        self.all_phi=[]
        self.all_rho=[]
        self.all_s=[]
        self.all_stabilities=[]
        self.all_physical=[]
        for i in np.linspace(0,1, grid_discretisation):
            for j in np.linspace(0,1-i, grid_discretisation):
                for k in np.linspace(0,1-i-j, grid_discretisation):
                    self.psi=np.array([[i,j],[k,1-i-j-k]])
                    self.run()
                    if not self.fixed_points or not np.any(np.all(np.round(self.psi, precision) == np.round(self.fixed_points, precision), axis=(1,2))):
                        self.fixed_points.append(self.psi)
                        self.update_observables()
                        self.all_phi.append(self.phi)
                        self.all_rho.append(self.rho)
                        self.all_s.append(self.s)
        for _ in range(num_random_search):
            self.psi=np.random.uniform(size=(2,2))
            self.run()
            if not self.fixed_points or not np.any(np.all(np.round(self.psi, precision) == np.round(self.fixed_points, precision), axis=(1,2))):
                self.fixed_points.append(self.psi)
                self.update_observables()
                self.all_phi.append(self.phi)
                self.all_rho.append(self.rho)
                self.all_s.append(self.s)
                self.all_stabilities.append(self.is_stable())
                self.all_physical.append(self.is_physical())
                
        if any(self.all_physical):
            self.phi=np.max(np.array(self.all_phi)[self.all_physical])
            idx_max_stable=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max_stable]
            self.s=self.all_s[idx_max_stable]
            self.psi=self.fixed_points[idx_max_stable]
            self.stability=self.all_stabilities[idx_max_stable]
            self.physical=True
        else:
            print("No physical fixed point found !")
            self.phi=max(self.all_phi)
            idx_max=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max]
            self.s=self.all_s[idx_max]
            self.psi=self.fixed_points[idx_max]
            self.stability=self.all_stabilities[idx_max]
            self.physical=False
                
                
    def find_all_fixed_points(self, grid_discretisation=20, precision=6, num_random_search=0, verbose=0):
        messages=[]
        for i in np.linspace(0,1,grid_discretisation):
            for j in np.linspace(0,1-i,grid_discretisation):
                for k in np.linspace(0,1-i-j,grid_discretisation):
                    messages.append(np.array([[i,j],[k, 1-i-j-k]]))
        for _ in range(num_random_search):
            messages.append(np.random.uniform(size=(2,2)))
        messages=np.array(messages)
        num_samples=grid_discretisation**3+num_random_search
        old_messages=None
        convergence=False
        
        for it in range(self.max_iter):
            old_messages=messages.copy()
            new_messages=np.zeros((num_samples,2,2))
            for i in range(2):
                for j in range(2):
                    for perm in self.permutations:
                        mult=np.ones(num_samples)
                        for k in perm:
                            mult*=messages[:,k,i]
                        new_messages[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
            Z=np.sum(new_messages, axis=(1,2))
            new_messages[np.where(Z==0)]=np.array([[0.25,0.25],[0.25,0.25]])
            Z=np.where(Z!=0,Z,1)
            new_messages/=Z.repeat(4).reshape(num_samples,2,2)
            messages=self.damping_parameter*messages+(1-self.damping_parameter)*new_messages
            if np.max(np.linalg.norm(messages-old_messages, axis=(1,2)))<self.tol:
                if verbose>=1:
                    print("Convergence reached for each starting initialisation after ", it, " steps !")
                convergence=True
                break

        if not convergence:
            print("Not every starting initialisation has converged after ", self.max_iter, " steps.")
                
        if verbose>=1:
            print('Max error: ', np.max(np.linalg.norm(messages-old_messages, axis=(1,2))))
            
        _, idx_unique=np.unique(np.round(messages, precision), axis=0, return_index=True)
        self.fixed_points=messages[idx_unique]
        self.all_phi=[]
        self.all_rho=[]
        self.all_s=[]
        self.all_stabilities=[]
        self.all_physical=[]
        for fixed_point in self.fixed_points:
            self.psi=fixed_point
            self.update_observables()
            self.all_phi.append(self.phi)
            self.all_rho.append(self.rho)
            self.all_s.append(self.s)
            self.all_stabilities.append(self.is_stable())
            self.all_physical.append(self.is_physical())
            
        if any(self.all_physical):
            self.phi=np.max(np.array(self.all_phi)[self.all_physical])
            idx_max_stable=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max_stable]
            self.s=self.all_s[idx_max_stable]
            self.psi=self.fixed_points[idx_max_stable]
            self.stability=self.all_stabilities[idx_max_stable]
            self.physical=True
        else:
            print("No physical fixed point found !")
            self.phi=max(self.all_phi)
            idx_max=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max]
            self.s=self.all_s[idx_max]
            self.psi=self.fixed_points[idx_max]
            self.stability=self.all_stabilities[idx_max]
            self.physical=False
        
    
    # More efficient for a grid bigger than ~40
    def find_all_fixed_points_torch(self, grid_discretisation=100, precision=6, num_random_search=0, verbose=0):
        messages=[]
        for i in np.linspace(0,1,grid_discretisation):
            for j in np.linspace(0,1-i,grid_discretisation):
                for k in np.linspace(0,1-i-j,grid_discretisation):
                    messages.append(np.array([[i,j],[k, 1-i-j-k]]))
        for _ in range(num_random_search):
            messages.append(np.random.uniform(size=(2,2)))
        messages=torch.from_numpy(np.array(messages)).to(self.device)
        num_samples=grid_discretisation**3+num_random_search
        old_messages=None
        convergence=False
        
        for it in range(self.max_iter):
            old_messages=messages.clone()
            new_messages=torch.zeros((num_samples,2,2))
            for i in range(2):
                for j in range(2):
                    for perm in self.permutations:
                        mult=torch.ones(num_samples)
                        for k in perm:
                            mult*=messages[:,k,i]
                        new_messages[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
            Z=torch.sum(new_messages, axis=(1,2))
            new_messages[torch.where(Z==torch.tensor([0.]))]=torch.tensor([[0.25,0.25],[0.25,0.25]])
            Z=torch.where(Z!=torch.tensor([0.]),Z,torch.tensor([1.]))
            new_messages/=Z.repeat_interleave(4).reshape(num_samples,2,2)
            messages=self.damping_parameter*messages+(1-self.damping_parameter)*new_messages
            if torch.max(torch.linalg.matrix_norm(messages-old_messages, dim=(1,2)))<self.tol:
                if verbose>=1:
                    print("Convergence reached for each starting initialisation after ", it, " steps !")
                convergence=True
                break


        if not convergence:
            print("Not every starting initialisation has converged after ", self.max_iter, " steps.")
                
        if verbose>=1:
            print('Max error: ', (torch.max(torch.linalg.norm(messages-old_messages, dim=(1,2)))).item())
        
        messages_np=messages.cpu().detach().numpy()
        _, idx_unique=np.unique(np.round(messages_np, precision), axis=0, return_index=True)
        self.fixed_points=messages_np[idx_unique]
        self.all_phi=[]
        self.all_rho=[]
        self.all_s=[]
        self.all_stabilities=[]
        self.all_physical=[]
        for fixed_point in self.fixed_points:
            self.psi=fixed_point
            self.update_observables()
            self.all_phi.append(self.phi)
            self.all_rho.append(self.rho)
            self.all_s.append(self.s)
            self.all_stabilities.append(self.is_stable())
            self.all_physical.append(self.is_physical())
            
        if any(self.all_physical):
            self.phi=np.max(np.array(self.all_phi)[self.all_physical])
            idx_max_stable=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max_stable]
            self.s=self.all_s[idx_max_stable]
            self.psi=self.fixed_points[idx_max_stable]
            self.stability=self.all_stabilities[idx_max_stable]
            self.physical=True
        else:
            print("No physical fixed point found for rule "+str(self.rule)+" !")
            self.phi=max(self.all_phi)
            idx_max=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max]
            self.s=self.all_s[idx_max]
            self.psi=self.fixed_points[idx_max]
            self.stability=self.all_stabilities[idx_max]
            self.physical=False
            
    def is_stable(self,fixed_point=None, precision=12, noise=1e-9, random_noise=False, num_tests=10 ):
        if fixed_point is None:
            fixed_point=self.psi
        old_psi=self.psi
        
        if random_noise:
            for i in range(num_tests):
                noisy_fixed_point=fixed_point+np.random.normal(0,noise,(2,2))
                noisy_fixed_point/=np.sum(noisy_fixed_point)
                self.psi=noisy_fixed_point
                self.run()
                if np.array_equal(np.round(self.psi,precision), np.round(fixed_point,precision)):
                    self.stability=True
                    self.psi=old_psi
                    if i>=1:
                        print("Warning: ", i+1, " stability checks were done before finding a stable noisy fixed point for rule ", self.rule, " .")
                    return True

            self.stability=False
            self.psi=old_psi
            return False
        else:
            noisy_fixed_point=fixed_point+noise*np.ones((2,2))
            noisy_fixed_point/=np.sum(noisy_fixed_point)
            self.psi=noisy_fixed_point
            self.run()
            if np.array_equal(np.round(self.psi,precision), np.round(fixed_point,precision)):
                self.stability=True
                self.psi=old_psi
                return True
            else:
                self.stability=False
                self.psi=old_psi
                return False
            
    
    def is_physical(self, fixed_point=None,  precision=10):
        if fixed_point is None:
            fixed_point=self.psi
        if np.array_equal(np.round(fixed_point), np.array([[0.,1],[0,0]])) or np.array_equal(np.round(fixed_point), np.array([[0.,0],[1,0]])):
            self.physical=False
            return False
        elif self.phi>np.log(2):
            print("Warning, entropy > log(2) even if fixed point physical !")
            self.physical=False
            return False
        else:
            self.physical=True
            return True

        
    


    
        
    
    
    