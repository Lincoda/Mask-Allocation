import numpy as np
import pandas as pd
from scipy.optimize import root

'''
Threeperiod_SIRD: describe the dynamic of SIRD and calculate number of death at period t=1 for resource constraints
evaluate_death: calculate number of deaths for SRA-I and II
evaluateT3: calculate state variables S and I under SRA-I and II mechanisms at period t=T
GRBT_evalDeath: calculate number of deaths for GRBT
GRBT_evalT3: calculate state variables S and I under GRBT mechanisms at period t=T
IC_cons: calculate gain from early participation for both groups of people
Udiff: calculate utility difference between people with and without mask initially

GRBTp1q0: return gain from early participation under fully separate senerio 
Mix_computeProb: given the shares of both groups of people who early participated in GRBT, compute the probs of receiving masks 
pmix_func: given the share of people without mask who early participated in GRBT, return their gain from early participation
qmix_func: given the share of people with mask who early participated in GRBT, return their gain from early participation
GRBT: given initial coverage rate m0, determine types of equilibrium and return number of deaths, probs of receiving masks, with other information
SRA1: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-I mechanism
SRA2: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-II mechanism

## consider case where q0!=q1 or q1!=q2
GRBTp1q0_growth: return gain from early participation under fully separate senerio
pmix_func_growth: given the share of people without mask who early participated in GRBT, return their gain from early participation
qmix_func_growth: given the share of people with mask who early participated in GRBT, return their gain from early participation
Mix_computeProb_growth: given the shares of both groups of people who early participated in GRBT, compute the probs of receiving masks
GRBT_growth: given initial coverage rate m0, determine types of equilibrium and return number of deaths, probs of receiving masks, with other information
SRA1_mchange: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-I mechanism
SRA2_mchange: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-II mechanism
'''

#-----------------------------------------------------------------------
# Complete Model 
#-----------------------------------------------------------------------

class Threeperiod_SIRD:
    
    def __init__(self,S=0.9,        # initial susceptible
                      I=0.1,        # initial infected
                      R=0,          # initial recovered
                      D=0,          # initial died
                      𝛽=2.4/(18/14),# basic transmission rate. R0=2.4 and it takes 18 days to leave I state in average.
                                    # Furthermore, a time unit is 14 days here.
                      𝛾=1-(17/18)**14,# propotion of people that will leave state I is one minus those does not leave in fourteen days 
                      𝛼=0.0138,     # propotion that will die after leave state I.
                                     
                      
                      T=0,          # model period
                      𝜋0=0.2,      # mask issued during period 0 

                      σo=0.5,       # old facemask inward protection
                      σn=0.7,       # new facemask inward protection
                      δo=0.5,       # old facemask outward protection
                      δn=0.7,       # new facemask outward protection

                      𝜋B=0.2,     # mask issued during period 1 for those who claim he does not own a mask during period 0
                      𝜋A=0.2,     # mask issued during period 1 for those who claim he owns a mask during period 0
                      
                      # (x,y) 
                      # x=0 if one claim he does not own a mask during period 0, x=1 otherwise 
                      # y=0 if one does not receive a mask during period 1, y=1 otherwise
                      𝜋B0=0.2,    # mask issued during period 2 for (0,0)
                      𝜋A0=0.2,    # mask issued during period 2 for (1,0) 
                      𝜋B1=0.2,    # mask issued during period 2 for (0,1) 
                      𝜋A1=0.2):   # mask issued during period 2 for (1,1) 

                                         
        self.S    = np.array([S,0,0,0,0,0,0,0])
        self.I    = np.array([I,0,0,0,0,0,0,0])
        self.R, self.D  = R, D
        self.𝛽, self.𝛾, self.𝛼 = 𝛽, 𝛾, 𝛼
        self.σo, self.σn, self.δo, self.δn = σo, σn, δo, δn

        self.T, self.𝜋0 = T, 𝜋0
        self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1 = πB, πA, πB0, πA0, πB1, πA1
        
    def evaluate_change(self):
        T = self.T
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        𝜋0, πB, πA, πB0, πA0, πB1, πA1 = self.𝜋0, self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1

        if T==0:
            # population distribution after issuing mask
            transition_0  = np.array([[1-𝜋0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [  𝜋0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(self.S) # 8x1
            I_mask = transition_0.dot(self.I) # 8x1

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            ##### compute number of deaths for resource constraints
            self.dB, self.dA = 𝛾 * 𝛼 * self.I[0], 𝛾* 𝛼 * self.I[1]
            # population distribution after issuing mask
            transition_1 =  np.array([[1-𝜋B,   0,0,0,0,0,0,0],
                                      [   0,1-𝜋A,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [  𝜋B,   0,0,0,0,0,0,0],
                                      [   0,  𝜋A,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0]])

                                      
            S_mask = transition_1.dot(self.S) # 8x1
            I_mask = transition_1.dot(self.I) # 8x1

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
            # population distribution after issuing mask
            # Because masks are sufficient after t=3, we don't need to track their history hereafter.
            transition_2 = np.array([[1-𝜋B0,0,0,0,0,1-𝜋A0,    0,    0],
                                     [    0,0,0,0,0,    0,1-𝜋B1,1-𝜋A1],
                                     [  𝜋B0,0,0,0,0,  𝜋A0,  𝜋B1,  𝜋A1]])
            S_mask = transition_2.dot(self.S) # 3x1
            I_mask = transition_2.dot(self.I) # 3x1

            # ϕ o n 
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T>=3:
            # Everyone can get a mask.
            transition = np.array([[0,0,0],
                                   [0,0,0],
                                   [1,1,1]])
            S_mask = transition.dot(self.S) # 3x1
            I_mask = transition.dot(self.I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        # Moving out from state I
        dR = 𝛾 * (1-𝛼) * I_mask # 3x1 vector 
        dD = 𝛾 * 𝛼 * I_mask # 3x1 vector 
        
        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = self.R + sum(dR)
        nD = self.D + sum(dD)

        # masking state transition after using.
        if T<=1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7
            
        else:
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])
        
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        return(np.array([nS,nI,nR,nD]))
    
    def update(self):
        
        change = self.evaluate_change()
        self.S, self.I, self.R, self.D = change
        self.T = self.T+1
    
    def generate_sequence(self, t):
        "Generate and return a time series of length t"
        S_path = [0]*t
        I_path = [0]*t
        R_path = [0]*t
        D_path = [0]*t
        
        for i in range(t):
            S_path[i], I_path[i], R_path[i], D_path[i]=sum(self.S), sum(self.I), self.R, self.D
            self.update()
        return(S_path,I_path,R_path,D_path)

    def severalupdates(self,t):
        for _ in range(t):
            self.update()
        return(self.D)



def evaluate_death(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=10,π0=0.2,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        # population distribution after issuing mask
        if T==0:
            transition_0  = np.array([[1-π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            # ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5
                                        [0,0,0,1,0,0,0,0],   # S3→S6
                                        [0,0,0,0,1,0,0,0]])  # S4→S7

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            transition_1 =  np.array([[1-πB,   0,0,0,0,0,0,0],
                                      [   0,1-πA,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [  πB,   0,0,0,0,0,0,0],
                                      [   0,  πA,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0]])

                                    
            S_mask = transition_1.dot(S) # 8x1
            I_mask = transition_1.dot(I) # 8x1

            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5
                                        [0,0,0,1,0,0,0,0],   # S3→S6
                                        [0,0,0,0,1,0,0,0]])  # S4→S7
            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            transition_2 = np.array([[1-πB0,0,0,0,0,1-πA0,    0,    0],
                                     [    0,0,0,0,0,    0,1-πB1,1-πA1],
                                     [  πB0,0,0,0,0,  πA0,  πB1,  πA1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T>=3:
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return(D)

def evaluateT3(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
               S=0.9,I=0.1,R=0,D=0,
               𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
               T=0,t=10,π0=0.2,
               σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
   
        if T==0:
            transition_0  = np.array([[1-π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
           
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7 

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            transition_1 =  np.array([[1-πB,   0,0,0,0,0,0,0],
                                      [   0,1-πA,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [  πB,   0,0,0,0,0,0,0],
                                      [   0,  πA,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0]])

                                    
            S_mask = transition_1.dot(S) # 8x1
            I_mask = transition_1.dot(I) # 8x1

           
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1 
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7 

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            
            transition_2 = np.array([[1-πB0,0,0,0,0,1-πA0,    0,    0],
                                     [    0,0,0,0,0,    0,1-πB1,1-πA1],
                                     [  πB0,0,0,0,0,  πA0,  πB1,  πA1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1 
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD
            
        elif T>=3:
            
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return sum(S),sum(I)



def GRBT_evalDeath(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=10,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        
        if T==0:
            transition_0  = np.array([[1-π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            
            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],[1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,1,0,0,0,0],   # S3→S2
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0], 
                                        [0,0,0,0,0,0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            signup= np.array([[1-p,0,  0,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  0,0,1-q,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  p,0,  0,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  0,0,  q,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)

            transition_1 =  np.array([[1-πA,0,   0,0,   0,0,   0,0],
                                      [  πA,0,   0,0,   0,0,   0,0],
                                      [   0,0,1-πA,0,   0,0,   0,0],
                                      [   0,0,  πA,0,   0,0,   0,0],
                                      [   0,0,   0,0,1-πB,0,   0,0],
                                      [   0,0,   0,0,  πB,0,   0,0],
                                      [   0,0,   0,0,   0,0,1-πB,0],
                                      [   0,0,   0,0,   0,0,  πB,0]])

                                    
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],[1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.eye(8)

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
         
            transition_2 = np.array([[1-πA0,    0,1-πA0,    0,1-πB0,    0,1-πB0,    0],
                                     [    0,1-πA1,    0,1-πA1,    0,1-πB1,    0,1-πB1],
                                     [  πA0,  πA1,  πA0,  πA1,  πB0,  πB1,  πB0,  πB1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
           
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
           
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T>=3:

            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1 
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return(D)

def GRBT_evalT3(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=10,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        if T==0:
            transition_0  = np.array([[1-π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            
            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],[1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1
            dD = 𝛾 * 𝛼 * I_mask # 3x1 
            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,1,0,0,0,0],   # S3→S2 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0], 
                                        [0,0,0,0,0,0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

          
            signup= np.array([[1-p,0,  0,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  0,0,1-q,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  p,0,  0,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  0,0,  q,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)

          
            transition_1 =  np.array([[1-πA,0,   0,0,   0,0,   0,0],
                                      [  πA,0,   0,0,   0,0,   0,0],
                                      [   0,0,1-πA,0,   0,0,   0,0],
                                      [   0,0,  πA,0,   0,0,   0,0],
                                      [   0,0,   0,0,1-πB,0,   0,0],
                                      [   0,0,   0,0,  πB,0,   0,0],
                                      [   0,0,   0,0,   0,0,1-πB,0],
                                      [   0,0,   0,0,   0,0,  πB,0]])

                                    
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],[1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1 
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            transition_mask = np.eye(8)

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            transition_2 = np.array([[1-πA0,    0,1-πA0,    0,1-πB0,    0,1-πB0,    0],
                                     [    0,1-πA1,    0,1-πA1,    0,1-πB1,    0,1-πB1],
                                     [  πA0,  πA1,  πA0,  πA1,  πB0,  πB1,  πB0,  πB1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
           
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
           
            dR = 𝛾 * (1-𝛼) * I_mask # 3x1 
            dD = 𝛾 * 𝛼 * I_mask # 3x1

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return sum(S),sum(I)

def IC_cons(v=0.5/0.7,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    phi_sign = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌
    phi_nsign= 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*𝜋A0*𝜌
    n_sign   = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*(v+𝜋B0*𝜌)
    n_nsign  = 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌)

    ICphi = phi_sign-phi_nsign
    ICn   = n_sign-n_nsign

    return ICphi, ICn

def Udiff(vo=0.5,vn=0.7,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    
    v=vo/vn
    phi_sign = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌
    phi_nsign= 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*𝜋A0*𝜌
    n_sign   = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*(v+𝜋B0*𝜌)
    n_nsign  = 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌)
    
    Uphi =    𝜌 * vn * max(phi_sign,phi_nsign)
    Un = vn + 𝜌 * vn * max(n_sign,n_nsign)
    
    return Uphi, Un, Un-Uphi


def GRBTp1q0(m=0.2,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛼=0.0138):

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    𝜋B_coef = (1-m)*(1-d0)
    𝜋A_coef =    m *(1-d0)
    𝜋B = min(m/𝜋B_coef,1)
    𝜋A = (m-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0

    
    if 𝜋B<1:
        𝜋2_0_coef=           m *(1-d0)-dA
        𝜋2_1_coef=(1-𝜋B)*((1-m)*(1-d0)-dB)
        𝜋2_2_coef=   𝜋B *((1-m)*(1-d0)-dB)
        
        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0

    else:
        𝜋2_0_coef=(1-𝜋A)*(m*(1-d0)-dA)
        𝜋2_1_coef=   𝜋A *(m*(1-d0)-dA)
        𝜋2_2_coef=       (1-m)*(1-d0)-dB

        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0

    ICphi, ICn = IC_cons(v=v,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
    
    return ICphi,ICn


def Mix_computeProb(x,y,m,I_0=0.05,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛼=0.0138):

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    # t=1 
    πB_coef=   x *(1-m)*(1-d0) +    y *m*(1-d0)
    πA_coef=(1-x)*(1-m)*(1-d0) + (1-y)*m*(1-d0)
    πB = min(m/πB_coef,1) 
    πA = (m-πB_coef)/πA_coef if πB==1 else 0

    # t=2 
    if πB<1:
        π2_0_coef=        (1-x)*((1-m)*(1-d0)-dB) + (1-y)*(m*(1-d0)-dA)
        π2_1_coef=(1-πB)*(   x *((1-m)*(1-d0)-dB) +    y *(m*(1-d0)-dA) )
        π2_2_coef=   πB *(   x *((1-m)*(1-d0)-dB) +    y *(m*(1-d0)-dA) )

        πA0 = min(m/π2_0_coef,1)
        πB0 = min((m-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*((1-x)*((1-m)*(1-d0)-dB) + (1-y)*(m*(1-d0)-dA) )
        π2_1_coef=   πA *((1-x)*((1-m)*(1-d0)-dB) + (1-y)*(m*(1-d0)-dA) )
        π2_2_coef=           x *((1-m)*(1-d0)-dB) +    y *(m*(1-d0)-dA)
        

        πA0 = min(m/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0
    
    return πB,πA,πA0,πB0,πA1,πB1

def pmix_func(x,m,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛼=0.0138):

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    # t=1 πB for those who sign up， πA for those who don't. We prioritize those who sign up over those who don't
    πB_coef=x*(1-d0)*(1-m)
    πA_coef=(1-x)*(1-d0)*(1-m)+(1-d0)*m
    πB = min(m/πB_coef,1)
    πA = (m-πB_coef)/πA_coef if πB==1 else 0

    # two cases at t=2
    if πB<1:
        π2_0_coef=        (1-x)*((1-m)*(1-d0)-dB) + m*(1-d0)-dA
        π2_1_coef= (1-πB)*   x *((1-m)*(1-d0)-dB)
        π2_2_coef=    πB *   x *((1-m)*(1-d0)-dB)


        πA0 = min(m/π2_0_coef,1)
        πB0 = min((m-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*( (1-x)*((1-m)*(1-d0)-dB) + m*(1-d0)-dA )
        π2_1_coef=   πA *( (1-x)*((1-m)*(1-d0)-dB) + m*(1-d0)-dA )
        π2_2_coef= x *((1-m)*(1-d0)-dB)

        πA0 = min(m/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    phi_sign = πB*(1+𝜌*(πB1+(1-πB1)*v)) + (1-πB)*πB0*𝜌
    phi_nsign= πA*(1+𝜌*(πA1+(1-πA1)*v)) + (1-πA)*πA0*𝜌

    return phi_sign-phi_nsign

def qmix_func(x,m,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛼=0.0138):

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    # t=1 πB for those who sign up， πA for those who don't. We prioritize those who sign up over those who don't
    πB_coef=(1-d0)*(1-m)+x*(1-d0)*m
    πA_coef=(1-x)*(1-d0)*m
    πB = min(m/πB_coef,1)
    πA = (m-πB_coef)/πA_coef if πB==1 else 0

    # two cases at t=2
    if πB<1:
        π2_0_coef= (1-x)*( m*(1-d0)-dA)
        π2_1_coef=(1-πB)*( (1-m)*(1-d0)-dB+ x*(m*(1-d0)-dA) )
        π2_2_coef=   πB *( (1-m)*(1-d0)-dB+ x*(m*(1-d0)-dA) )

        πA0 = min(m/π2_0_coef,1)
        πB0 = min((m-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*(1-x)*( m*(1-d0)-dA)
        π2_1_coef=   πA *(1-x)*( m*(1-d0)-dA)
        π2_2_coef= (1-m)*(1-d0)-dB+ x*(m*(1-d0)-dA) 
        
        πA0 = min(m/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    n_sign = πB*(1+𝜌*(πB1+(1-πB1)*v)) + (1-πB)*(v+πB0*𝜌)
    n_nsign= πA*(1+𝜌*(πA1+(1-πA1)*v)) + (1-πA)*(v+πA0*𝜌)
    return n_sign-n_nsign

def GRBT(m=0.2,I_0=0.05,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,vo=0.5,vn=0.7,𝛼=0.0138):
    
    ICphi_sep, ICn_sep = GRBTp1q0(m=m,I_0=I_0,v=vo/vn,𝜌=𝜌,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    
    #### Fully-Separating Equilibrium ####
    if ICphi_sep>=0 and ICn_sep<=0:
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb(1,0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
        D_val = GRBT_evalDeath(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
        S3,I3 = GRBT_evalT3(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,1,0,ICphi_sep,ICn_sep,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    
    if ICphi_sep<0:
        #### Partial-Separating Equilibrium. #### 
        # People without mask play mix strategy
        p_res=root(pmix_func,1,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n,𝛼),method='broyden1',tol=10e-12)

        if p_res.success and p_res.x>0 and p_res.x<1:
            p_star=p_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb(x=p_star,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
            D_val = GRBT_evalDeath(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
            S3,I3 = GRBT_evalT3(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
            ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            
            return m,p_star,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        
        #### Pooling Equlibrium ####
        # No people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb(x=0,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
        D_val = GRBT_evalDeath(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
        S3,I3 = GRBT_evalT3(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
        ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud

        
        
    if ICn_sep>0:
        #### Partial-Separating Equilibrium. #### 
        # People having mask play mix strategy
        q_res=root(qmix_func,0.2,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n,𝛼),method='broyden1',tol=10e-12)

        if q_res.success and q_res.x>0 and q_res.x<1:
            q_star=q_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb(x=1,y=q_star,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
            D_val = GRBT_evalDeath(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
            S3,I3 = GRBT_evalT3(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
            ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)

            return m,1,q_star,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        
        #### Pooling Equlibrium ####
        # All people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb(x=1,y=1,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
        D_val = GRBT_evalDeath(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
        S3,I3 = GRBT_evalT3(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
        ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
def SRA1(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7,vo=0.5,vn=0.7,𝜌=1,𝛼=0.0138):

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    π1 = m/(1-d0)

    π2_coef = (1-m)*(1-d0)-dB + m*(1-d0)-dA

    π2 = m/π2_coef

    func = evaluate_death(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2,
                          σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
    S3,I3 = evaluateT3(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2,
                       σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
    Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2)

    return {'func':func,'π1':π1,'π2':π2,'S3':S3,'I3':I3,'Uphi':Uphi,'Un':Un,'Ud':Ud}
    
def SRA2(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7,vo=0.5,vn=0.7,𝜌=1,𝛼=0.0138):

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    π1 = m/(1-d0)

    π20_coef = (1-π1)*( (1-m)*(1-d0)-dB + m*(1-d0)-dA )
    π21_coef =    π1 *( (1-m)*(1-d0)-dB + m*(1-d0)-dA )

    π20 = min(m/π20_coef,1)
    π21 = (m-π20_coef)/π21_coef if π20==1 else 0

    func = evaluate_death(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21,
                          σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
    S3,I3 = evaluateT3(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21,
                       σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
    Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21)

    return {'func':func,'π1':π1,'π20':π20,'π21':π21,'S3':S3,'I3':I3,'Uphi':Uphi,'Un':Un,'Ud':Ud}
    
    
    
    
    
    
    

def GRBTp1q0_growth(m=0.2,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='hop',𝛼=0.0138):
    
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m0,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    𝜋B_coef = (1-m0)*(1-d0)
    𝜋A_coef =    m0 *(1-d0)
    𝜋B = min(m1/𝜋B_coef,1)
    𝜋A = (m1-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0

    
    if 𝜋B<1:
        𝜋2_0_coef=           m0 *(1-d0)-dA
        𝜋2_1_coef=(1-𝜋B)*((1-m0)*(1-d0)-dB)
        𝜋2_2_coef=   𝜋B *((1-m0)*(1-d0)-dB)
        
        𝜋A0 = min(m2/𝜋2_0_coef,1)
        𝜋B0 = min((m2-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m2-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0

    else:
        𝜋2_0_coef=(1-𝜋A)*(m0*(1-d0)-dA)
        𝜋2_1_coef=   𝜋A *(m0*(1-d0)-dA)
        𝜋2_2_coef=       (1-m0)*(1-d0)-dB

        𝜋A0 = min(m2/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m2-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m2-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0

    ICphi, ICn = IC_cons(v=v,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
    
    return ICphi,ICn
      
    
def pmix_func_growth(x,m,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='stock',𝛼=0.0138):

    # growth pattern
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m
        
    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m0,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    # t=1 πB for those who sign up， πA for those who don't. We prioritize those who sign up over those who don't
    πB_coef=x*(1-d0)*(1-m0)
    πA_coef=(1-x)*(1-d0)*(1-m0)+(1-d0)*m0
    πB = min(m1/πB_coef,1)
    πA = (m1-πB_coef)/πA_coef if πB==1 else 0

    # two cases at t=2
    if πB<1:
        π2_0_coef=        (1-x)*((1-m0)*(1-d0)-dB) + m0*(1-d0)-dA
        π2_1_coef= (1-πB)*   x *((1-m0)*(1-d0)-dB)
        π2_2_coef=    πB *   x *((1-m0)*(1-d0)-dB)


        πA0 = min(m2/π2_0_coef,1)
        πB0 = min((m2-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*( (1-x)*((1-m0)*(1-d0)-dB) + m0*(1-d0)-dA )
        π2_1_coef=   πA *( (1-x)*((1-m0)*(1-d0)-dB) + m0*(1-d0)-dA )
        π2_2_coef= x *((1-m0)*(1-d0)-dB)

        πA0 = min(m2/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m2-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    phi_sign = πB*(1+𝜌*(πB1+(1-πB1)*v)) + (1-πB)*πB0*𝜌
    phi_nsign= πA*(1+𝜌*(πA1+(1-πA1)*v)) + (1-πA)*πA0*𝜌

    return phi_sign-phi_nsign


def qmix_func_growth(x,m,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='stock',𝛼=0.0138):

    # growth pattern
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m
        
    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m0,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA
        
    # t=1 πB for those who sign up， πA for those who don't. We prioritize those who sign up over those who don't
    πB_coef=(1-d0)*(1-m0)+x*(1-d0)*m0
    πA_coef=(1-x)*(1-d0)*m0
    πB = min(m1/πB_coef,1)
    πA = (m1-πB_coef)/πA_coef if πB==1 else 0

    # two cases at t=2
    if πB<1:
        π2_0_coef= (1-x)*( m0*(1-d0)-dA)
        π2_1_coef=(1-πB)*( (1-m0)*(1-d0)-dB+ x*(m0*(1-d0)-dA) )
        π2_2_coef=   πB *( (1-m0)*(1-d0)-dB+ x*(m0*(1-d0)-dA) )

        πA0 = min(m2/π2_0_coef,1)
        πB0 = min((m2-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*(1-x)*( m0*(1-d0)-dA)
        π2_1_coef=   πA *(1-x)*( m0*(1-d0)-dA)
        π2_2_coef= (1-m0)*(1-d0)-dB+ x*(m0*(1-d0)-dA) 
        
        πA0 = min(m2/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m2-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    n_sign = πB*(1+𝜌*(πB1+(1-πB1)*v)) + (1-πB)*(v+πB0*𝜌)
    n_nsign= πA*(1+𝜌*(πA1+(1-πA1)*v)) + (1-πA)*(v+πA0*𝜌)

    return n_sign-n_nsign


def Mix_computeProb_growth(x,y,m,I_0=0.05,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='stock',𝛼=0.0138):

    # growth pattern
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m
                           
    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m0,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    # t=1 
    πB_coef=   x *(1-m0)*(1-d0) +    y *m0*(1-d0)
    πA_coef=(1-x)*(1-m0)*(1-d0) + (1-y)*m0*(1-d0)
    πB = min(m1/πB_coef,1)
    πA = (m1-πB_coef)/πA_coef if πB==1 else 0

    # t=2 
    if πB<1:
        π2_0_coef=        (1-x)*((1-m0)*(1-d0)-dB) + (1-y)*(m0*(1-d0)-dA)
        π2_1_coef=(1-πB)*(   x *((1-m0)*(1-d0)-dB) +    y *(m0*(1-d0)-dA) )
        π2_2_coef=   πB *(   x *((1-m0)*(1-d0)-dB) +    y *(m0*(1-d0)-dA) )

        πA0 = min(m2/π2_0_coef,1)
        πB0 = min((m2-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*((1-x)*((1-m0)*(1-d0)-dB) + (1-y)*(m0*(1-d0)-dA) )
        π2_1_coef=   πA *((1-x)*((1-m0)*(1-d0)-dB) + (1-y)*(m0*(1-d0)-dA) )
        π2_2_coef=           x *((1-m0)*(1-d0)-dB) +    y *(m0*(1-d0)-dA)
        

        πA0 = min(m2/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m2-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0
    
    return πB,πA,πA0,πB0,πA1,πB1

def GRBT_growth(m=0.2,I_0=0.05,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='hop',vo=0.5,vn=0.7,𝛼=0.0138):
    
    # growth pattern
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m
        
    ICphi_sep, ICn_sep = GRBTp1q0_growth(m=m,I_0=I_0,v=vo/vn,𝜌=𝜌,σo=σo,σn=σn,δo=δo,δn=δn,pattern=pattern,𝛼=𝛼)
    
    # Fully-Separating Equilibrium
    if ICphi_sep>=0 and ICn_sep<=0:
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_growth(1,0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,pattern=pattern,𝛼=𝛼)
        D_val = GRBT_evalDeath(p=1,q=0,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
        S3,I3 = GRBT_evalT3(p=1,q=0,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m0,m1,m2,1,0,ICphi_sep,ICn_sep,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    if ICphi_sep<0:
        #### Partial-Separating Equilibrium. #### 
        # People without mask play mix strategy
        p_res=root(pmix_func_growth,0.95,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n,pattern,𝛼),method='broyden1',tol=10e-12)

        if p_res.success and p_res.x>0 and p_res.x<1:
            p_star=p_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_growth(x=p_star,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,pattern=pattern,𝛼=𝛼)
            D_val = GRBT_evalDeath(p=p_star,q=0,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
            S3,I3 = GRBT_evalT3(p=p_star,q=0,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
            ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            
            return m0,m1,m2,p_star,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        #### Pooling Equlibrium ####
        # People having mask play mix strategy
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_growth(x=0,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,pattern=pattern,𝛼=𝛼)
        D_val = GRBT_evalDeath(p=0,q=0,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
        S3,I3 = GRBT_evalT3(p=0,q=0,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
        ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m0,m1,m2,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
    if ICn_sep>0:
        #### Partial-Separating Equilibrium. #### 
        # People having mask play mix strategy
        q_res=root(qmix_func_growth,0.1,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n,pattern,𝛼),method='df-sane',tol=10e-12)

        if q_res.success and q_res.x>0 and q_res.x<1:
            q_star=q_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_growth(x=1,y=q_star,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,pattern=pattern,𝛼=𝛼)
            D_val = GRBT_evalDeath(p=1,q=q_star,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
            S3,I3 = GRBT_evalT3(p=1,q=q_star,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
            ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            
            return m0,m1,m2,1,q_star,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        #### Pooling Equlibrium ####
        # All people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_growth(x=1,y=1,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,pattern=pattern,𝛼=𝛼)
        D_val = GRBT_evalDeath(p=1,q=1,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
        S3,I3 = GRBT_evalT3(p=1,q=1,π0=m0,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
        ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m0,m1,m2,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud

def SRA1_mchange(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='hop',vo=0.5,vn=0.7):
    
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m0,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    π1 = m1/(1-d0)

    π2_coef = (1-m0)*(1-d0)-dB + m0*(1-d0)-dA

    π2 = m2/π2_coef

    func = evaluate_death(S=1-I_0,I=I_0,π0=m0,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2,
                          σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    S3,I3 = evaluateT3(S=1-I_0,I=I_0,π0=m0,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2,
                       σo=σo,σn=σn,δo=δo,δn=δn,t=3)
    Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=1,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2)

    return {'func':func,'π1':π1,'π2':π2,'S3':S3,'I3':I3,'Uphi':Uphi,'Un':Un,'Ud':Ud}

def SRA2_mchange(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7,pattern='hop',vo=0.5,vn=0.7,𝛼=0.0138):
    
    if pattern=='hop':
        m0,m1,m2=m,m,1.15*m
    elif pattern=='stock':
        m0,m1,m2=1.15*m,m,m

    # number of deaths at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,π0=m0,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    π1 = m1/(1-d0)

    π20_coef = (1-π1)*( (1-m0)*(1-d0)-dB + m0*(1-d0)-dA )
    π21_coef =    π1 *( (1-m0)*(1-d0)-dB + m0*(1-d0)-dA )

    π20 = min(m2/π20_coef,1)
    π21 = (m2-π20_coef)/π21_coef if π20==1 else 0

    func = evaluate_death(S=1-I_0,I=I_0,π0=m0,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21,
                          σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=300)
    S3,I3 = evaluateT3(S=1-I_0,I=I_0,π0=m0,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21,
                       σo=σo,σn=σn,δo=δo,δn=δn,𝛼=𝛼,t=3)
    Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=1,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21)

    return {'func':func,'π1':π1,'π20':π20,'π21':π21,'S3':S3,'I3':I3,'Uphi':Uphi,'Un':Un,'Ud':Ud}



