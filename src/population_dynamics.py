from src.BP import *

import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import optimize

homo_messages=[np.array([[1.,0],[0,0]]), np.array([[0., 0],[0,1]]), np.array([[0.5, 0.5],[0,0]]), np.array([[0,0],[0.5,0.5]])]

class AN:
    def __init__(self, rule, mu=0):
        self.rule=rule
        self.BP=BP_OT(rule, mu=mu)
        self.SP=population_dynamics_OT_torch(rule, mu=mu)
        self.phase=None
        self.solutions=None
    def __repr__(self):
        description=''
        if self.phase is not None:
            description+='Phase: '+self.phase
            description+='\nSolution(s): '+self.solutions+'\n'
        return description+str(self.BP)+'\n\n'+str(self.SP)
    
    def RS_and_1RSB(self,tol=1e-3, tol_BP=1e-10):
        self.BP.find_all_fixed_points_torch()
        if self.BP.physical==True:
            self.SP.run()
        
        
        number_homogeneous_configurations=0
        if self.rule[0]=='0' or self.rule[0]=='+':
            number_homogeneous_configurations+=1
        if self.rule[-1]=='1' or self.rule[-1]=='+':
            number_homogeneous_configurations+=1
        
        isnan=False
        if self.BP.physical:
            if np.isnan(self.SP.phi_mean) or np.isneginf(self.SP.phi_mean):
                print("Probably should not be here !")
                isnan=True
            
        if isnan or self.BP.physical==False or abs(self.BP.phi-self.SP.phi_mean)<tol:
            self.phase='RS'
            if self.BP.phi<-tol_BP:
                self.solutions='No stationary solutions'
            elif np.abs(self.BP.phi)<tol_BP:
                num_homo=0
                num_phi_0=0
                for j in range(len(self.BP.all_phi)):
                    if abs(self.BP.all_phi[j])<tol_BP:
                        num_phi_0+=1
                    if np.any(np.all(np.round(self.BP.fixed_points[j],1) == homo_messages, axis=(1,2))):
                        num_homo+=1
                if num_homo!=number_homogeneous_configurations:
                    print("Some homogeneous configuration was not found by BP for rule ", self.rule)
                if num_homo==num_phi_0:
                    self.solutions='Only homogeneous stationary solutions'
                else:
                    if num_phi_0>num_homo and num_homo>0:
                        self.solutions='Subexponentially many with homogeneous stationary solutions'
                    elif num_phi_0>num_homo and num_homo==0:
                        self.solutions='Subexponentially many with no homogeneous stationary solutions'
                    else:
                        print(num_phi_0)
                        print(num_homo)
                        print("You should not see this message !")
            elif self.BP.phi>tol_BP:
                if number_homogeneous_configurations>0:
                    self.solutions="Exponentially many with homogeneous stationary solutions "
                else:
                    self.solutions="Exponentially many with no homogeneous stationary solutions "

            
                
        elif self.BP.phi>self.SP.phi_mean:
            if self.SP.complexity>-tol:
                self.phase='d1RSB'
                if self.SP.psi_mean<-tol:
                    self.solutions='No stationary solutions'
                elif np.abs(self.SP.psi_mean)<tol:
                    self.solutions='Subexponentially many stationary solutions, no informations on homogeneous'
                elif self.SP.psi_mean>tol:
                    self.solutions='Exponentially many solutions, no informations on homogeneous'
            else:
                self.phase='s1RSB'
                self.SP.compute_static_transition()
                if self.SP.phi_s<0:
                    self.solutions='No stationary solutions'
                elif np.abs(self.SP.phi_s)<tol:
                    self.solutions='Subexponentially many stationary solutions, no informations on homogeneous'
                elif self.SP.phi_s>tol:
                    self.solutions='Exponentially many solutions, no informations on homogeneous'

            
        if self.BP.physical and self.BP.phi+tol<self.SP.phi_mean:
            print("Warning... BP entropy smaller than SP entropy for rule ",self.rule, ", computing static 1RSB and hoping that this solves the problem.")
            self.phase='s1RSB'
            self.SP.compute_static_transition()
            if self.SP.phi_s>self.BP.phi+tol:
                print("Warning: the BP entropy is still smaller that the SP entropy even after static 1RSB calculation !")
            if self.SP.phi_s<0:
                self.solutions='No stationary solutions'
            elif np.abs(self.SP.phi_s)<tol:
                self.solutions='Subexponentially many stationary solutions, no informations on homogeneous'
            elif self.SP.phi_s>tol:
                self.solutions='Exponentially many solutions, no informations on homogeneous'
            
            
            

# This class is deprecated
class population_dynamics_OT_not_vectorized:
    def __init__(self, rule, m=1, M=10000, num_samples=20000, hard_fields=True, damping=True, damping_parameter=0.8, mu=0, rng=np.random.default_rng()):
        self.rule=rule
        self.m=m
        self.M=M
        self.num_samples=num_samples
        self.hard_fields=hard_fields
        self.damping=damping
        self.damping_parameter=0.8
        self.mu=mu
        self.rng=rng

        self.d=len(rule)-1
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d-1)))
        self.nb_new_samples=round((1-self.damping_parameter)*self.M)   

        
        self.population=np.zeros((M,2,2))
        if self.hard_fields:
            self.population=self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]]), self.M)
        else:
            self.population=rng.uniform(size=(M,2,2))
        self.old_population=self.population
        self.weights=np.ones(M)/M
        self.old_weights=self.weights
        
        self.psi=None
        self.phi=None
        self.complexity=None        

      
        
    def respect_rule(self,i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
    
    def BP_one_step(self,psi):
        psi_new=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=1
                    for l, k in enumerate(perm):
                        mult*=psi[l,k,i]
                    psi_new[i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
        Z=np.sum(psi_new)
        if Z!=0:
            psi_new=psi_new/Z
        else:
            psi_new=np.array([[0.25, 0.25],[0.25, 0.25]])
            
        return psi_new, Z
    
    def sample_messages(self):
        indexes=np.random.choice(self.M, self.d-1, p=self.old_weights)
        V_old=self.population[indexes]
        return V_old
    
    def step(self):
        self.old_population=self.population.copy()
        self.old_weights=self.weights.copy()
        p=np.random.permutation(self.M)
        self.population=self.population[p]
        self.weights=self.weights[p]
        if self.damping==False:
            for i in range(M):
                old_messages=self.sample_messages()
                self.population[i], self.weights[i]=self.BP_one_step(old_messages)
        else:
            for i in range(self.nb_new_samples):
                    old_messages=self.sample_messages()
                    self.population[i], self.weights[i]=self.BP_one_step(old_messages)
        
        sum_weights=np.sum(self.weights)
        if sum_weights==0:
            self.weights=np.ones(self.M)/self.M
        else:
            self.weights=self.weights/sum_weights
        
    
    def compute_Z_i(self, pop_sample):
        Z_i=0
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=pop_sample[0,j,i]
                    for l, k in enumerate(perm):
                        mult*=pop_sample[l+1,k,i]
                    Z_i+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
        return Z_i
    
    def compute_Z_ij(self, pop_sample):
        Z_ij=0
        for i in range(2):
            for j in range(2):
                Z_ij+=pop_sample[0,i,j]*pop_sample[1,j,i]
        return Z_ij
    
    
    def update_observables(self, num_samples=None):
        if num_samples==None:
            num_samples=self.num_samples
        Z_i=0
        Z_i_deriv=0
        Z_ij=0
        Z_ij_deriv=0
        for _ in range(num_samples):
            indices=np.random.randint(0,self.M,size=self.d)
            _i=self.compute_Z_i(self.population[indices])
            power=np.power(_i,self.m)
            Z_i+=power
            Z_i_deriv+=power*np.log(_i) if _i !=0.0 else 0.0

            indices=np.random.randint(0,self.M,size=2)
            _ij=self.compute_Z_ij(self.population[indices])
            power=np.power(_ij,self.m)
            Z_ij+=power
            Z_ij_deriv+=power*np.log(_ij) if _ij !=0.0 else 0.0

        Z_i=Z_i/num_samples
        Z_i_deriv=Z_i_deriv/num_samples

        Z_ij=Z_ij/num_samples
        Z_ij_deriv=Z_ij_deriv/num_samples
        
        self.psi=np.log(Z_i)-self.d/2*np.log(Z_ij)
        self.phi=Z_i_deriv/Z_i-self.d/2*Z_ij_deriv/Z_ij
        self.complexity=self.psi-self.m*self.phi
        
    def diff(self, n_bins=1000):    
        old_hist_00=np.histogram(self.old_population[:,0,0], bins=n_bins, range=(0,1), density=True)
        old_hist_01=np.histogram(self.old_population[:,0,1], bins=n_bins, range=(0,1), density=True)
        old_hist_10=np.histogram(self.old_population[:,1,0], bins=n_bins, range=(0,1), density=True)
        old_hist_11=np.histogram(self.old_population[:,1,1], bins=n_bins, range=(0,1), density=True)
        new_hist_00=np.histogram(self.population[:,0,0], bins=n_bins, range=(0,1), density=True)
        new_hist_01=np.histogram(self.population[:,0,1], bins=n_bins, range=(0,1), density=True)
        new_hist_10=np.histogram(self.population[:,1,0], bins=n_bins, range=(0,1), density=True)
        new_hist_11=np.histogram(self.population[:,1,1], bins=n_bins, range=(0,1), density=True)
        return np.sum(np.abs(new_hist_00[0]-old_hist_00[0])+np.abs(new_hist_01[0]-old_hist_01[0])+np.abs(new_hist_10[0]-old_hist_10[0])+np.abs(new_hist_11[0]-old_hist_11[0]))
        

    def draw_population(self,n_bins=100, title=None, old=False):
        f, ax=plt.subplots(2,2)
        f.suptitle(title)
        xlabels=[[r'$\psi_{0,0}$',r'$\psi_{0,1}$'],[r'$\psi_{1,0}$',r'$\psi_{1,1}$']]
        for i in range(2):
            for j in range(2):
                if not old:
                    pop=self.population[:,i,j]
                else:
                    pop=self.old_population[:,i,j]
                ax[i,j].hist(pop, bins=n_bins, range=(0,1), density=True)
                ax[i,j].set_xlabel(xlabels[i][j])
                ax[i,j].set_ylabel('Approximated distribution')
                ax[i,j].set_xlim((0,1))
            
        plt.tight_layout()
        
        
        
#===============================================================================================
# This class has not all the required functionality. Please use the class population_dynamics_OT_torch instead
class population_dynamics_OT:
    def __init__(self, rule, m=1, mu=0, M=10000, num_samples=100000, damping_parameter=0.8, hard_fields=True, planted_messages=None, fraction_planted_messages=0.9, impose_symmetry=False, rng=np.random.default_rng()):
        self.rule=rule
        self.m=m
        self.M=M
        self.num_samples=num_samples
        self.hard_fields=hard_fields
        self.impose_symmetry=impose_symmetry
        self.damping_parameter=0.8
        self.mu=mu
        self.rng=rng

        self.d=len(rule)-1
        self.d_min_1=self.d-1
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d_min_1)))
        self.nb_new_samples=round((1-self.damping_parameter)*self.M)   

        
        self.population=np.zeros((M,2,2))
        if self.hard_fields:
            self.population=self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]]), self.M)
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.planted_messages[:round(self.M*self.fraction_planted_messages)] 
        else:
            self.population=rng.uniform(size=(M,2,2))
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.plantent_messages[:round(self.M*self.fraction_planted_messages)]
                
        self.old_population=self.population
        
        self.no_update=False
        
        self.psi=None
        self.phi=None
        self.complexity=None
        
        self.rho=None
        self.s=None

      
        
    def respect_rule(self,i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
    
    def BP_one_step(self,psi):
        psi_new=np.zeros((self.nb_new_samples,2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=np.ones(self.nb_new_samples)
                    for l, k in enumerate(perm):
                        mult*=psi[:,l,k,i]
                    psi_new[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
        Z=np.sum(psi_new, axis=(1,2))
        pow_Z=np.power(Z,self.m)
        sum_pow_Z=np.sum(pow_Z)
        if sum_Z!=0:
            indexes=np.random.choice(self.nb_new_samples, self.nb_new_samples, p=pow_Z/sum_pow_Z)
            if self.impose_symmetry:
                mid_index=round(self.nb_new_samples/2)
                psi_new=np.concatenate(np.flip(psi_new[indexes[:mid_index]],psi_new[indexes[mid_index:]], axis=(1,2)))
            else:
                psi_new=psi_new[indexes]
            Z=Z[indexes]
            psi_new/=np.repeat(Z, repeats=4).reshape(self.nb_new_samples,2,2)
            self.population[np.random.choice(self.M, self.nb_new_samples)]=psi_new
            self.no_update=False
        else:
            self.no_update=True
        
            
    def step(self):
        if self.no_update==False:
            self.old_population=self.population.copy()
        
        indexes=np.random.choice(self.M, self.nb_new_samples*(self.d_min_1))
        old_messages=self.population[indexes].reshape(self.nb_new_samples,self.d_min_1,2,2)
        self.BP_one_step(old_messages)
        
    
    
    def update_observables(self, num_samples=None):
        if num_samples==None:
            num_samples=self.num_samples

        indices=np.random.randint(0,self.M,size=num_samples*self.d)
        pop_sample_Z_i=self.population[indices].reshape((num_samples,self.d,2,2))
        indices=np.random.randint(0,self.M,size=num_samples*2)
        pop_sample_Z_ij=self.population[indices].reshape((num_samples,2,2,2))
        
        _i=np.zeros(num_samples)
        _ij=np.zeros(num_samples)
        _i_prime=torch.zeros(num_samples)
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=pop_sample_Z_i[:,0,j,i].copy()
                    for l, k in enumerate(perm):
                        mult*=pop_sample_Z_i[:,l+1,k,i]
                    _i+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    _i_prime+=i*np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    
                _ij+=pop_sample_Z_ij[:,0,i,j]*pop_sample_Z_ij[:,1,j,i]
        power=np.power(_i,self.m)
        Z_i=np.sum(power)
        Z_i_deriv=np.sum(power*np.where(_i!=0.,np.log(_i),0.))

        power=np.power(_ij,self.m)
        Z_ij=np.sum(power)
        Z_ij_deriv=np.sum(power*np.where(_ij!=0,np.log(_ij),0))

        Z_i=Z_i/num_samples
        Z_i_deriv=Z_i_deriv/num_samples

        Z_ij=Z_ij/num_samples
        Z_ij_deriv=Z_ij_deriv/num_samples

        self.psi=np.log(Z_i)-self.d/2*np.log(Z_ij)
        self.phi=Z_i_deriv/Z_i-self.d/2*Z_ij_deriv/Z_ij
        self.complexity=self.psi-self.m*self.phi
        self.rho=(1/Z_i*torch.sum(_i_prime*torch.pow(_i,self.m-1))/num_samples).item()
        self.s=self.phi-self.mu*self.rho
        
        

        
    def diff(self, n_bins=1000):    
        old_hist_00=np.histogram(self.old_population[:,0,0], bins=n_bins, range=(0,1), density=True)
        old_hist_01=np.histogram(self.old_population[:,0,1], bins=n_bins, range=(0,1), density=True)
        old_hist_10=np.histogram(self.old_population[:,1,0], bins=n_bins, range=(0,1), density=True)
        old_hist_11=np.histogram(self.old_population[:,1,1], bins=n_bins, range=(0,1), density=True)
        new_hist_00=np.histogram(self.population[:,0,0], bins=n_bins, range=(0,1), density=True)
        new_hist_01=np.histogram(self.population[:,0,1], bins=n_bins, range=(0,1), density=True)
        new_hist_10=np.histogram(self.population[:,1,0], bins=n_bins, range=(0,1), density=True)
        new_hist_11=np.histogram(self.population[:,1,1], bins=n_bins, range=(0,1), density=True)
        return np.sum(np.abs(new_hist_00[0]-old_hist_00[0])+np.abs(new_hist_01[0]-old_hist_01[0])+np.abs(new_hist_10[0]-old_hist_10[0])+np.abs(new_hist_11[0]-old_hist_11[0]))
        

    def draw_population(self,n_bins=100, title=None, old=False):
        f, ax=plt.subplots(2,2)
        f.suptitle(title)
        xlabels=[[r'$\psi_{0,0}$',r'$\psi_{0,1}$'],[r'$\psi_{1,0}$',r'$\psi_{1,1}$']]
        for i in range(2):
            for j in range(2):
                if not old:
                    pop=self.population[:,i,j]
                else:
                    pop=self.old_population[:,i,j]
                ax[i,j].hist(pop, bins=n_bins, range=(0,1), density=True)
                ax[i,j].set_xlabel(xlabels[i][j])
                ax[i,j].set_ylabel('Approximated distribution')
                ax[i,j].set_xlim((0,1))
            
        plt.tight_layout()
        
    def fraction_dont_care(self):
        return np.sum(np.where(np.count_nonzero(self.population, dim=(1,2))==4,1,0))/self.M

#====================================================================================================================================================================

def fitting_func(x,a,b,c,d):
    return a+b*np.power(2,x)+c*np.power(3,x)

import torch



class population_dynamics_OT_torch:
    def __init__(self, rule, m=1, mu=0, M=2000000, num_samples=40000000, damping_parameter=0.8, hard_fields=True, planted_messages=None, fraction_planted_messages=0.9, impose_symmetry=False, max_iter=10000, tol=1000, convergence_check_interval=200, sampling_threshold=8000, sampling_interval=50, m_list=np.linspace(0.00001,1, 30), rng=np.random.default_rng(), device='cuda'):
        if device=='cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.rule=rule
        self.m=m
        self.mu=mu
        self.M=M
        self.num_samples=num_samples
        self.damping_parameter=0.8
        self.hard_fields=hard_fields
        self.planted_messages=planted_messages
        self.fraction_planted_messages=fraction_planted_messages
        self.impose_symmetry=impose_symmetry
        self.max_iter=max_iter
        self.tol=tol
        self.convergence_check_interval=convergence_check_interval
        self.sampling_threshold=sampling_threshold
        self.sampling_interval=sampling_interval
        self.rng=rng
        self.device=device

        self.d=len(rule)-1
        self.d_min_1=self.d-1
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d_min_1)))
        self.nb_new_samples=round((1-self.damping_parameter)*self.M)   
        self.d_min_1_times_num_samples=self.nb_new_samples*self.d_min_1

        
        self.population=np.zeros((M,2,2))
        if self.hard_fields:
            self.population=torch.from_numpy(self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]], dtype=np.float32), self.M)).to(self.device)
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.planted_messages[:round(self.M*self.fraction_planted_messages)]                                                                    
                #self.planted_message.repeat((np.round(self.M*self.fraction_planted_messages),1,1))
        else:
            self.population=torch.from_numpy(rng.uniform(size=(M,2,2))).to(self.device)
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.plantent_messages[:round(self.M*self.fraction_planted_messages)]                                                                    
        
        self.old_population=self.population
        
        self.no_update=False
        
        self.phi=None
        self.complexity=None          
        self.psi=None 
        self.rho=None
        self.s=None
        
        self.phi_mean=None
        self.complexity_mean=None
        self.psi_mean=None
        self.rho_mean=None
        self.s_mean=None
        
        self.phi_std=None
        self.complexity_std=None
        self.psi_std=None
        self.rho_std=None
        self.s_std=None
        
        self.m_list=m_list
        self.phi_list=None
        self.phi_list_std=None
        self.complexity_list=None
        self.complexity_list_std=None
        self.psi_list=None
        self.psi_list_std=None
        self.rho_list=None
        self.rho_list_std=None
        self.s_list=None
        self.s_list_std=None
        
        self.fitting_param_phi=None
        self.fitting_param_rho=None
        
        self.rho_s=None
        self.rho_d=None
        self.phi_s=None
        self.phi_d=None
    
    def __repr__(self):
        description="Instance of class \'population_dynamics_OT_torch\'\nRule : "+str(self.rule)+"\nμ =  "+str(self.mu)+"\nPopulation size: "+str(self.M)
        if self.phi_mean is not None:
            description+='\nφ = '+str(self.phi_mean)+' +/- '+str(self.phi_std)
            description+='\nΣ = '+str(self.complexity_mean)+' +/- '+str(self.complexity_std)
            description+='\nΨ = '+str(self.psi_mean)+' +/- '+str(self.psi_std)
            description+='\nρ = '+str(self.rho_mean)+' +/- '+str(self.rho_std)
            description+='\ns = '+str(self.s_mean)+' +/- '+str(self.s_std)
        return description

           
    def respect_rule(self,i, j, rest_config):
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
        if self.no_update==False:
             self.old_population=self.population.clone()           
        psi=self.population[torch.randint(self.M, (self.d_min_1_times_num_samples,))].reshape(self.nb_new_samples,self.d_min_1,2,2)
        psi_new=torch.zeros((self.nb_new_samples,2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=torch.ones(self.nb_new_samples)
                    for l, k in enumerate(perm):
                        mult*=psi[:,l,k,i]
                    psi_new[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
        Z=torch.sum(psi_new, axis=(1,2))
        sum_Z=torch.sum(Z)
        if sum_Z!=0:
            indexes=torch.multinomial(torch.pow(Z,self.m), self.nb_new_samples, replacement=True)
            if self.impose_symmetry:
                mid_index=round(self.nb_new_samples/2)
                psi_new=torch.cat((psi_new[indexes[:mid_index]],psi_new[indexes[mid_index:]].flip(dims=(1,2))))
            else:
                psi_new=psi_new[indexes]
            Z=Z[indexes]
            psi_new/=Z.repeat_interleave(4).reshape(self.nb_new_samples,2,2)
            self.population[torch.randint(self.M, (self.nb_new_samples,))]=psi_new
            self.no_update=False
        else:
            self.no_update=True

    
    def run(self, max_iter=None, tol=None, check_convergence=True, convergence_check_interval=None, sampling_threshold=None, sampling_interval=None, reset_population=False, verbose=0):
        if reset_population:
            self.reset_population
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        if convergence_check_interval is None:
            convergence_check_interval=self.convergence_check_interval
        if sampling_threshold is None:
            sampling_threshold=self.sampling_threshold
        if sampling_interval is None:
            sampling_interval=self.sampling_interval
        
        phi_list=[]
        complexity_list=[]
        psi_list=[]
        rho_list=[]
        s_list=[]
        for i in range(max_iter):
            self.step()
            if check_convergence and i%convergence_check_interval==0:
                diff=self.diff()
                if verbose>=2:
                    print('Difference after ', i, 'iterations : ', diff)
                if diff<tol:
                    if verbose>=1:
                        print('Early stopping ! Convergence of ', tol, 'reached after ', i, 'iterations.')
                    break
            if i%sampling_interval==0 and i>=sampling_threshold:
                self.update_observables()
                phi_list.append(self.phi)
                complexity_list.append(self.complexity)
                psi_list.append(self.psi)
                rho_list.append(self.rho)
                s_list.append(self.s)
        self.step()
        self.update_observables()
        phi_list.append(self.phi)
        complexity_list.append(self.complexity)
        psi_list.append(self.psi)
        rho_list.append(self.rho)
        s_list.append(self.s)
        if verbose>=1:
            print("Finished ! Final difference: ", self.diff())
        self.phi_mean, self.phi_std=np.mean(phi_list), np.std(phi_list)
        self.complexity_mean, self.complexity_std=np.mean(complexity_list), np.std(complexity_list)
        self.psi_mean, self.psi_std=np.mean(psi_list), np.std(psi_list)
        self.rho_mean, self.rho_std=np.mean(rho_list), np.std(rho_list)
        self.s_mean, self.s_std=np.mean(s_list), np.std(s_list)

                   
    def update_observables(self, num_samples=None):
        if num_samples==None:
            num_samples=self.num_samples

        pop_sample_Z_i=self.population[torch.randint(self.M,(num_samples*self.d,))].reshape((num_samples,self.d,2,2))
        pop_sample_Z_ij=self.population[torch.randint(self.M,(num_samples*2,))].reshape((num_samples,2,2,2))
        
        _i=torch.zeros(num_samples)
        _ij=torch.zeros(num_samples)
        _i_prime=torch.zeros(num_samples)
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=pop_sample_Z_i[:,0,j,i].clone()
                    for l, k in enumerate(perm):
                        mult*=pop_sample_Z_i[:,l+1,k,i]
                    _i+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    _i_prime+=i*np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult

                _ij+=pop_sample_Z_ij[:,0,i,j]*pop_sample_Z_ij[:,1,j,i]
                
        power=torch.pow(_i,self.m)
        Z_i=torch.sum(power)
        Z_i_deriv=torch.sum(power*torch.where(_i!=torch.tensor([0.]),torch.log(_i),torch.tensor([0.])))
        
        power=torch.pow(_ij,self.m)
        Z_ij=torch.sum(power)
        Z_ij_deriv=torch.sum(power*torch.where(_ij!=torch.tensor([0.]),torch.log(_ij),torch.tensor([0.])))

        Z_i=Z_i/num_samples
        Z_i_deriv=Z_i_deriv/num_samples

        Z_ij=Z_ij/num_samples
        Z_ij_deriv=Z_ij_deriv/num_samples


        self.psi=(torch.log(Z_i)-self.d/2*torch.log(Z_ij)).item()
        self.phi=(Z_i_deriv/Z_i-self.d/2*Z_ij_deriv/Z_ij).item()
        self.complexity=self.psi-self.m*self.phi
        
        self.rho=(1/Z_i*torch.sum(_i_prime*torch.where(_i!=torch.tensor([0.], dtype=torch.float),torch.pow(_i,self.m-1), torch.tensor([0.], dtype=torch.float)))/num_samples).item()
        self.s=self.phi-self.mu*self.rho
        
    def reset_population(self):
        self.population=np.zeros((self.M,2,2))
        if self.hard_fields:
            self.population=torch.from_numpy(self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]], dtype=np.float32), self.M)).to(self.device)
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.planted_messages[:round(self.M*self.fraction_planted_messages)]
        else:
            self.population=torch.from_numpy(rng.uniform(size=(M,2,2))).to(self.device)
            if self.planted_messages is not None:
                                self.population[:round(self.M*self.fraction_planted_messages)]=self.plantent_messages[:round(self.M*self.fraction_planted_messages)]                                                                    

    
    def compute_static_transition(self, m_list=None, verbose=0):
        if m_list is not None:
            self.m_list=m_list
        
        num_m=len(self.m_list)
        self.phi_list=np.zeros(num_m)
        self.phi_list_std=np.zeros(num_m)
        self.complexity_list=np.zeros(num_m)
        self.complexity_list_std=np.zeros(num_m)
        self.psi_list=np.zeros(num_m)
        self.psi_list_std=np.zeros(num_m)
        self.rho_list=np.zeros(num_m)
        self.rho_list_std=np.zeros(num_m)
        self.s_list=np.zeros(num_m)
        self.s_list_std=np.zeros(num_m)
        
        for i, m in enumerate(self.m_list):
            if verbose>=1:
                print('========== m = ', m, ' ==========')
            self.m=m
            self.run(check_convergence=False, reset_population=True, verbose=verbose)
            self.phi_list[i]=self.phi_mean
            self.complexity_list[i]=self.complexity_mean
            self.psi_list[i]=self.psi_mean
            self.rho_list[i]=self.rho_mean
            self.s_list[i]=self.s_mean
            self.phi_list_std[i]=self.phi_std
            self.complexity_list_std[i]=self.complexity_std
            self.rho_list_std[i]=self.rho_std
            self.s_list_std[i]=self.s_std
            
        if np.max(self.complexity_list)<0:
            if verbose>=1:
                print("No intersection with the complexity=0 line !")
            self.phi_s=np.max(self.phi_list)
            return
                    
        index_max=np.argmax(self.complexity_list)
        
        self.fitting_param_phi, _ = optimize.curve_fit(fitting_func, self.phi_list[index_max:], self.complexity_list[index_max:], method="lm")
        phi_samples=np.linspace(min(self.phi_list),max(self.phi_list),100000)
        self.phi_s=phi_samples[np.argmin(np.abs(fitting_func(phi_samples, *self.fitting_param_phi)))]
        self.phi_d=self.phi_list[index_max]
        
        self.fitting_param_rho, _ = optimize.curve_fit(fitting_func, self.rho_list[index_max:], self.complexity_list[index_max:], method="lm")
        rho_samples=np.linspace(min(self.rho_list),max(self.rho_list),100000)
        self.rho_s=rho_samples[np.argmin(np.abs(fitting_func(rho_samples, *self.fitting_param_rho)))]
        self.rho_d=self.rho_list[index_max]
        
        
    def fraction_dont_care(self):
        return (torch.sum(torch.where(torch.count_nonzero(self.population, dim=(1,2))==4,1,0))/self.M).item()
        
    def diff(self, n_bins=1000):    
        old_hist_00=torch.histc(self.old_population[:,0,0], bins=n_bins, min=0, max=0)
        old_hist_01=torch.histc(self.old_population[:,0,1], bins=n_bins, min=0, max=0)
        old_hist_10=torch.histc(self.old_population[:,1,0], bins=n_bins, min=0, max=0)
        old_hist_11=torch.histc(self.old_population[:,1,1], bins=n_bins, min=0, max=0)
        new_hist_00=torch.histc(self.population[:,0,0], bins=n_bins, min=0, max=0)
        new_hist_01=torch.histc(self.population[:,0,1], bins=n_bins, min=0, max=0)
        new_hist_10=torch.histc(self.population[:,1,0], bins=n_bins, min=0, max=0)
        new_hist_11=torch.histc(self.population[:,1,1], bins=n_bins, min=0, max=0)
        return (torch.sum(torch.abs(new_hist_00-old_hist_00)+torch.abs(new_hist_01-old_hist_01)+torch.abs(new_hist_10-old_hist_10)+torch.abs(new_hist_11-old_hist_11))).item()
        

    def draw_population(self,n_bins=100, title=None, old=False):
        f, ax=plt.subplots(2,2)
        f.suptitle(title)
        xlabels=[[r'$\psi_{0,0}$',r'$\psi_{0,1}$'],[r'$\psi_{1,0}$',r'$\psi_{1,1}$']]
        for i in range(2):
            for j in range(2):
                if not old:
                    pop=self.population.cpu().numpy()[:,i,j]
                else:
                    pop=self.old_population.cpu().numpy()[:,i,j]
                ax[i,j].hist(pop, bins=n_bins, range=(0,1), density=True)
                ax[i,j].set_xlabel(xlabels[i][j])
                ax[i,j].set_ylabel('Approximated distribution')
                ax[i,j].set_xlim((0,1))
            
        #plt.tight_layout()
        
    def draw_sigma_phi(self, errorbars=False, title=None):
        if title is None:
            title="$\mu = $"+str(self.mu)
            
        f, ax=plt.subplots()
        ax.set_title(title)
        
        ax.axhline(0, linestyle='--', color='k', markersize=1)
        ax.axvline(self.phi_s, linestyle='--', color='grey', markersize=1)

        samples=np.linspace(min(self.phi_list),max(self.phi_list),1000)
        ax.plot(samples, fitting_func(samples, *self.fitting_param_phi))
        ax.plot(self.phi_list, self.complexity_list, 'x')
        
        if errorbars:
            ax.errorbar(self.phi_list, self.complexity_list, xerr=self.phi_list_std, yerr=self.complexity_list_std, fmt='.', capsize=3)
            
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\Sigma$');
            
    def draw_sigma_rho(self, errorbars=False, title=None):
        if title is None:
            title="$\mu = $"+str(self.mu)
            
        f, ax=plt.subplots()
        ax.set_title(title)
        
        ax.axhline(0, linestyle='--', color='k', markersize=1)
        ax.axvline(self.rho_s, linestyle='--', color='grey', markersize=1)

        samples=np.linspace(min(self.rho_list),max(self.rho_list),1000)
        ax.plot(samples, fitting_func(samples, *self.fitting_param_rho))
        ax.plot(self.rho_list, self.complexity_list, 'x')
        
        if errorbars:
            ax.errorbar(self.rho_list, self.complexity_list, xerr=self.rho_list_std, yerr=self.complexity_list_std, fmt='.', capsize=3)
            
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$\Sigma$');
           
        
                

        

        
        
