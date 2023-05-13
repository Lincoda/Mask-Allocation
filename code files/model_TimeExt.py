import numpy as np
import pandas as pd
from scipy.optimize import root


'''
T4_SIRD: describe the dynamic of SIRD and calculate number of death at period t=1,2  for resource constraints
evalD_EXT: calculate number of deaths for SRA-I and II
evalT4_EXT: calculate state variables S and I under SRA-I and II mechanisms at period t=T
GRBT_evalD_EXT: calculate number of deaths for GRBT
GRBT_evalT4_EXT: calculate state variables S and I under GRBT mechanisms at period t=T
IC_cons_EXT: calculate gain from early participation for both groups of people
Udiff_EXT: calculate utility difference between people with and without mask initially

GRBTp1q0_EXT: return gain from early participation under fully separate senerio 
pmix_func_EXT: given the share of people without mask who early participated in GRBT, return their gain from early participation
qmix_func_EXT: given the share of people with mask who early participated in GRBT, return their gain from early participation
Mix_computeProb_EXT: given the shares of both groups of people who early participated in GRBT, compute the probs of receiving masks 
GRBT_EXT: given initial coverage rate m0, determine types of equilibrium and return number of deaths, probs of receiving masks, with other information

SRA1_EXT: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-I mechanism
SRA2_EXT: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-II mechanism
'''





class T4_SIRD:
    
    def __init__(self,S=0.99,        # initial susceptible
                      I=0.01,        # initial infected
                      R=0,           # initial recovered
                      D=0,           # initial died
                      𝛽=10*2.4/18, # basic transmission rate. R0=2.4 and it takes 18 days to leave I state in average.
                                     # Furthermore, a time unit is 10 days here.
                      𝛾=1-(17/18)**10,# propotion of people that will leave state I is one minus those does not leave in ten days 
                      𝛼=0.0138,      # propotion that will die after leave state I.
                                     
                      
                      T=0,          # model period
                      𝜋0=0.2,      # mask issued during period 0 

                      σo=0.5,       # old facemask inward protection
                      σn=0.7,       # new facemask inward protection
                      δo=0.5,       # old facemask outward protection
                      δn=0.7,       # new facemask outward protection

                      πB=0.2,     # mask issued during period 1 for those who claim he does not own a mask during period 0
                      πA=0.2,     # mask issued during period 1 for those who claim he owns a mask during period 0
                      
                      # (x,y) 
                      # x=0 if one claim he does not own a mask during period 0, x=1 otherwise 
                      # y=0 if one does not receive a mask during period 1, y=1 otherwise
                      πB0=0.2,    # mask issued during period 2 for (0,0)
                      πA0=0.2,    # mask issued during period 2 for (1,0) 
                      πB1=0.2,    # mask issued during period 2 for (0,1) 
                      πA1=0.2,    # mask issued during period 2 for (1,1)
                 
                      πB00=0.2,
                      πB01=0.2,
                      πA00=0.2,
                      πA01=0.2,
                      πB10=0.2,
                      πB11=0.2,
                      πA10=0.2,
                      πA11=0.2):   

                                         
        self.S    = np.array([S,0,0,0,0,0,0,0])
        self.I    = np.array([I,0,0,0,0,0,0,0])
        self.R, self.D  = R, D
        self.𝛽, self.𝛾, self.𝛼 = 𝛽, 𝛾, 𝛼
        self.σo, self.σn, self.δo, self.δn = σo, σn, δo, δn

        self.T, self.𝜋0 = T, 𝜋0
        self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1 = πB, πA, πB0, πA0, πB1, πA1
        self.πB00, self.πB01, self.πB10, self.πB11 = πB00, πB01, πB10, πB11
        self.πA00, self.πA01, self.πA10, self.πA11 = πA00, πA01, πA10, πA11
        
    def evaluate_change(self):
        T = self.T
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        𝜋0, πB, πA, πB0, πA0, πB1, πA1 = self.𝜋0, self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1
        πB00, πB01, πB10, πB11 = self.πB00, self.πB01, self.πB10, self.πB11 
        πA00, πA01, πA10, πA11 = self.πA00, self.πA01, self.πA10, self.πA11

        if T==0:
            # population distribution after issuing mask
            transition_0  = np.array([[1-𝜋0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  𝜋0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(self.S) # 8x1
            I_mask = transition_0.dot(self.I) # 8x1

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            ##### compute numbers of death for resource constraints.
            self.dB, self.dA = 𝛾 * 𝛼 * self.I[0], 𝛾* 𝛼 * self.I[1]
            # population distribution after issuing mask
            transition_1 =  np.array([[1-πB,   0,0,0,0,0,0,0],
                                      [   0,1-πA,0,0,0,0,0,0],
                                      [   0,   0,0,0,0,0,0,0],
                                      [  πB,   0,0,0,0,0,0,0],
                                      [   0,  πA,0,0,0,0,0,0],
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
            ##### compute numbers of death for resource constraints.
            self.dB0, self.dB1, self.dA0, self.dA1 = 𝛾 * 𝛼 * self.I[0], 𝛾* 𝛼 * self.I[6],𝛾* 𝛼 * self.I[5], 𝛾* 𝛼 * self.I[7]
            # population distribution after issuing mask
            transition_2 = np.array([ [1-πB0,0,0,0,0,    0,    0,    0],
                                      [  πB0,0,0,0,0,    0,    0,    0],
                                      [    0,0,0,0,0,1-πA0,    0,    0],
                                      [    0,0,0,0,0,  πA0,    0,    0],
                                      [    0,0,0,0,0,    0,1-πB1,    0],
                                      [    0,0,0,0,0,    0,  πB1,    0],
                                      [    0,0,0,0,0,    0,    0,1-πA1],
                                      [    0,0,0,0,0,    0,    0,  πA1]])
            
            S_mask = transition_2.dot(self.S) # 3x1
            I_mask = transition_2.dot(self.I) # 3x1

            # masking state: ϕ n ϕ n o n o n
            matrix = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)],[1,(1-σn),1,(1-σn),(1-σo),(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T==3:
            # population distribution after issuing mask
            transition_3 = np.array([[1-πB00,     0,1-πA00,     0,1-πB10,     0,1-πA10,     0],
                                     [     0,1-πB01,     0,1-πA01,     0,1-πB11,     0,1-πA11],
                                     [  πB00,  πB01,  πA00,  πA01,  πB10,  πB11,  πA10,  πA11]])
            
            S_mask = transition_3.dot(self.S) # 3x1
            I_mask = transition_3.dot(self.I) # 3x1

            # masking state: ϕ o n
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T>=4:
            transition = np.array([[0,0,0],
                                   [0,0,0],
                                   [1,1,1]])
            S_mask = transition.dot(self.S) # 3x1
            I_mask = transition.dot(self.I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        # moving out from state I
        dR = 𝛾 * (1-𝛼) * I_mask 
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = self.R + sum(dR)
        nD = self.D + sum(dD)

        if T<=1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7 
            
        elif T==2:
            transition_mask = np.identity(8)
            
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
    

def evalD_EXT(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
             πB00=0.2, πB01=0.2, πB10=0.2, πB11=0.2,
             πA00=0.2, πA01=0.2, πA10=0.2, πA11=0.2,
             S=0.99,I=0.01,R=0,D=0,
             𝛽=10*2.4/18,𝛾=1-(17/18)**10,𝛼=0.0138,
             T=0,t=10,π0=0.2,
             σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    
    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        if T==0:
            # population distribution after issuing mask
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

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            # population distribution after issuing mask
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

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
            # population distribution after issuing mask
            transition_2 = np.array([ [1-πB0,0,0,0,0,    0,    0,    0],
                                      [  πB0,0,0,0,0,    0,    0,    0],
                                      [    0,0,0,0,0,1-πA0,    0,    0],
                                      [    0,0,0,0,0,  πA0,    0,    0],
                                      [    0,0,0,0,0,    0,1-πB1,    0],
                                      [    0,0,0,0,0,    0,  πB1,    0],
                                      [    0,0,0,0,0,    0,    0,1-πA1],
                                      [    0,0,0,0,0,    0,    0,  πA1]])
            
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            # masking state: ϕ n ϕ n o n o n
            matrix = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)],[1,(1-σn),1,(1-σn),(1-σo),(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T==3:
            # population distribution after issuing mask
            transition_3 = np.array([[1-πB00,     0,1-πA00,     0,1-πB10,     0,1-πA10,     0],
                                     [     0,1-πB01,     0,1-πA01,     0,1-πB11,     0,1-πA11],
                                     [  πB00,  πB01,  πA00,  πA01,  πB10,  πB11,  πA10,  πA11]])
            
            S_mask = transition_3.dot(S) # 3x1
            I_mask = transition_3.dot(I) # 3x1

            # masking state: ϕ o n
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T>=4:

            transition = np.array([[0,0,0],
                                   [0,0,0],
                                   [1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        # SIRD dynamics
        dR = 𝛾 * (1-𝛼) * I_mask 
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = R + sum(dR)
        nD = D + sum(dD)
        
        if T<=1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7
            
        elif T==2:
            transition_mask = np.identity(8)
            
        else:
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])
            
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        
        S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return(D)



def evalT4_EXT(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
             πB00=0.2, πB01=0.2, πB10=0.2, πB11=0.2,
             πA00=0.2, πA01=0.2, πA10=0.2, πA11=0.2,
             S=0.99,I=0.01,R=0,D=0,
             𝛽=10*2.4/18,𝛾=1-(17/18)**10,𝛼=0.0138,
             T=0,t=3,π0=0.2,
             σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    
    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):

        if T==0:
            # population distribution after issuing mask
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

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            # population distribution after issuing mask
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

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
            # population distribution after issuing mask
            transition_2 = np.array([ [1-πB0,0,0,0,0,    0,    0,    0],
                                      [  πB0,0,0,0,0,    0,    0,    0],
                                      [    0,0,0,0,0,1-πA0,    0,    0],
                                      [    0,0,0,0,0,  πA0,    0,    0],
                                      [    0,0,0,0,0,    0,1-πB1,    0],
                                      [    0,0,0,0,0,    0,  πB1,    0],
                                      [    0,0,0,0,0,    0,    0,1-πA1],
                                      [    0,0,0,0,0,    0,    0,  πA1]])
            
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            # masking state: ϕ n ϕ n o n o n
            matrix = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)],[1,(1-σn),1,(1-σn),(1-σo),(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T==3:
            # population distribution after issuing mask
            transition_3 = np.array([[1-πB00,     0,1-πA00,     0,1-πB10,     0,1-πA10,     0],
                                     [     0,1-πB01,     0,1-πA01,     0,1-πB11,     0,1-πA11],
                                     [  πB00,  πB01,  πA00,  πA01,  πB10,  πB11,  πA10,  πA11]])
            
            S_mask = transition_3.dot(S) # 3x1
            I_mask = transition_3.dot(I) # 3x1

            # masking state: ϕ o n
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            

        
        # moving out from state I
        dR = 𝛾 * (1-𝛼) * I_mask 
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = R + sum(dR)
        nD = D + sum(dD)
        
        if T<=1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7
            
        elif T==2:
            transition_mask = np.identity(8)
            
        else:
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])
            
            
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        
        S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return sum(S),sum(I)

def GRBT_evalD_EXT(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   πB00=0.2, πB01=0.2, πB10=0.2, πB11=0.2,
                   πA00=0.2, πA01=0.2, πA10=0.2, πA11=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=10*2.4/18,𝛾=1-(17/18)**10,𝛼=0.0138,
                   T=0,t=10,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        if T==0:
            # population distribution after issuing mask
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

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            
            signup = np.array([[1-p,  0,0,0,0,0,0,0],
                               [  0,1-q,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0],
                               [  p,  0,0,0,0,0,0,0],
                               [  0,  q,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)
            
            # population distribution after issuing mask
            transition_1 =  np.array([[1-πA,   0,0,0,   0,   0,0,0],
                                      [   0,1-πA,0,0,   0,   0,0,0],
                                      [  πA,   0,0,0,   0,   0,0,0],
                                      [   0,  πA,0,0,   0,   0,0,0],
                                      [   0,   0,0,0,1-πB,   0,0,0],
                                      [   0,   0,0,0,   0,1-πB,0,0],
                                      [   0,   0,0,0,  πB,   0,0,0],
                                      [   0,   0,0,0,   0,  πB,0,0]])

                                      
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            # masking state: ϕ o n n ϕ o n n
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),1,(1-δo),(1-δn),(1-δn)],[1,(1-σo),(1-σn),(1-σn),1,(1-σo),(1-σn),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
            # population distribution after issuing mask
            transition_2 = np.array([ [1-πA0,1-πA0,    0,    0,    0,    0,    0,    0],
                                      [  πA0,  πA0,    0,    0,    0,    0,    0,    0],
                                      [    0,    0,1-πA1,1-πA1,    0,    0,    0,    0],
                                      [    0,    0,  πA1,  πA1,    0,    0,    0,    0],
                                      [    0,    0,    0,    0,1-πB0,1-πB0,    0,    0],
                                      [    0,    0,    0,    0,  πB0,  πB0,    0,    0],
                                      [    0,    0,    0,    0,    0,    0,1-πB1,1-πB1],
                                      [    0,    0,    0,    0,    0,    0,  πB1,  πB1]])
            
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            # masking state: ϕ n o n ϕ n o n o o
            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],
                              [1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T==3:
            # population distribution after issuing mask
            transition_3 = np.array([[1-πA00,     0,1-πA10,     0,1-πB00,     0,1-πB10,     0],
                                     [     0,1-πA01,     0,1-πA11,     0,1-πB01,     0,1-πB11],
                                     [  πA00,  πA01,  πA10,  πA11,  πB00,  πB01,  πB10,  πB11]])
            
            S_mask = transition_3.dot(S) # 3x1
            I_mask = transition_3.dot(I) # 3x1

            # masking state: ϕ o n
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T>=4:
            transition = np.array([[0,0,0],
                                   [0,0,0],
                                   [1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1
            
            # masking state: ϕ o n
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            

        
        # moving out from state I
        dR = 𝛾 * (1-𝛼) * I_mask 
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = R + sum(dR)
        nD = D + sum(dD)
        
        if T<1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6 
                                        [0,0,0,0,1,0,0,0]])  # S4→S7
            
        elif T==1 or T==2:
            transition_mask = np.identity(8)
            
        else:
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])
            
            
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        
        S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return(D)

def GRBT_evalT4_EXT(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   πB00=0.2, πB01=0.2, πB10=0.2, πB11=0.2,
                   πA00=0.2, πA01=0.2, πA10=0.2, πA11=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=10*2.4/18,𝛾=1-(17/18)**10,𝛼=0.0138,
                   T=0,t=10,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        if T==0:
            # population distribution after issuing mask
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

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            
            signup = np.array([[1-p,  0,0,0,0,0,0,0],
                               [  0,1-q,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0],
                               [  p,  0,0,0,0,0,0,0],
                               [  0,  q,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0],
                               [  0,  0,0,0,0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)
            
            # population distribution after issuing mask
            transition_1 =  np.array([[1-πA,   0,0,0,   0,   0,0,0],
                                      [   0,1-πA,0,0,   0,   0,0,0],
                                      [  πA,   0,0,0,   0,   0,0,0],
                                      [   0,  πA,0,0,   0,   0,0,0],
                                      [   0,   0,0,0,1-πB,   0,0,0],
                                      [   0,   0,0,0,   0,1-πB,0,0],
                                      [   0,   0,0,0,  πB,   0,0,0],
                                      [   0,   0,0,0,   0,  πB,0,0]])

                                      
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            # masking state: ϕ o n n ϕ o n n
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),1,(1-δo),(1-δn),(1-δn)],[1,(1-σo),(1-σn),(1-σn),1,(1-σo),(1-σn),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
            # population distribution after issuing mask
            transition_2 = np.array([ [1-πA0,1-πA0,    0,    0,    0,    0,    0,    0],
                                      [  πA0,  πA0,    0,    0,    0,    0,    0,    0],
                                      [    0,    0,1-πA1,1-πA1,    0,    0,    0,    0],
                                      [    0,    0,  πA1,  πA1,    0,    0,    0,    0],
                                      [    0,    0,    0,    0,1-πB0,1-πB0,    0,    0],
                                      [    0,    0,    0,    0,  πB0,  πB0,    0,    0],
                                      [    0,    0,    0,    0,    0,    0,1-πB1,1-πB1],
                                      [    0,    0,    0,    0,    0,    0,  πB1,  πB1]])
            
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            # masking state: ϕ n o n ϕ n o n o o
            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],
                              [1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T==3:
            # population distribution after issuing mask
            transition_3 = np.array([[1-πA00,     0,1-πA10,     0,1-πB00,     0,1-πB10,     0],
                                     [     0,1-πA01,     0,1-πA11,     0,1-πB01,     0,1-πB11],
                                     [  πA00,  πA01,  πA10,  πA11,  πB00,  πB01,  πB10,  πB11]])
            
            S_mask = transition_3.dot(S) # 3x1
            I_mask = transition_3.dot(I) # 3x1

            # masking state: ϕ o n
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            

        
        # moving out from state I
        dR = 𝛾 * (1-𝛼) * I_mask 
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = R + sum(dR)
        nD = D + sum(dD)
        
        if T<1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 
                                        [0,0,0,1,0,0,0,0],   # S3→S6
                                        [0,0,0,0,1,0,0,0]])  # S4→S7
            
        elif T==1 or T==2:
            transition_mask = np.identity(8)
        else:
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])
            
            
            
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        
        S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return sum(S),sum(I)




def IC_cons_EXT(v=0.5,𝜌=1,πB=0.2,
                πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                πB00=0.2, πB01=0.2, πB10=0.2, πB11=0.2,
                 πA00=0.2, πA01=0.2, πA10=0.2, πA11=0.2):
    
    P_A3 = np.array([𝜋A*𝜋A1*𝜋A11,
                     𝜋A*𝜋A1*(1-𝜋A11),
                     𝜋A*(1-𝜋A1)*𝜋A10,
                     𝜋A*(1-𝜋A1)*(1-𝜋A10),
                     (1-𝜋A)*𝜋A0*𝜋A01,
                     (1-𝜋A)*𝜋A0*(1-𝜋A01),
                     (1-𝜋A)*(1-𝜋A0)*𝜋A00,
                     (1-𝜋A)*(1-𝜋A0)*(1-𝜋A00)])
    P_B3 = np.array([𝜋B*𝜋B1*𝜋B11,
                     𝜋B*𝜋B1*(1-𝜋B11),
                     𝜋B*(1-𝜋B1)*𝜋B10,
                     𝜋B*(1-𝜋B1)*(1-𝜋B10),
                     (1-𝜋B)*𝜋B0*𝜋B01,
                     (1-𝜋B)*𝜋B0*(1-𝜋B01),
                     (1-𝜋B)*(1-𝜋B0)*𝜋B00,
                     (1-𝜋B)*(1-𝜋B0)*(1-𝜋B00)])
    v_B3 = np.array([1+𝜌+𝜌**2,1+𝜌+𝜌**2*v,1+𝜌*v+𝜌**2,1+𝜌*v,𝜌+𝜌**2,𝜌+𝜌**2*v,𝜌**2,0])
    v_A3 = np.array([0,0,0,0,v,v,v,v])+v_B3 
    

    ICphi = P_B3.dot(v_B3) - P_A3.dot(v_B3)
    ICn   = P_B3.dot(v_A3) - P_A3.dot(v_A3)
    return ICphi, ICn

def Udiff_EXT(vo=0.5,vn=0.7,𝜌=1,πB=0.2,
                πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                πB00=0.2, πB01=0.2, πB10=0.2, πB11=0.2,
                 πA00=0.2, πA01=0.2, πA10=0.2, πA11=0.2):
    
    v=vo/vn
    P_A3 = np.array([𝜋A*𝜋A1*𝜋A11,
                     𝜋A*𝜋A1*(1-𝜋A11),
                     𝜋A*(1-𝜋A1)*𝜋A10,
                     𝜋A*(1-𝜋A1)*(1-𝜋A10),
                     (1-𝜋A)*𝜋A0*𝜋A01,
                     (1-𝜋A)*𝜋A0*(1-𝜋A01),
                     (1-𝜋A)*(1-𝜋A0)*𝜋A00,
                     (1-𝜋A)*(1-𝜋A0)*(1-𝜋A00)])
    P_B3 = np.array([𝜋B*𝜋B1*𝜋B11,
                     𝜋B*𝜋B1*(1-𝜋B11),
                     𝜋B*(1-𝜋B1)*𝜋B10,
                     𝜋B*(1-𝜋B1)*(1-𝜋B10),
                     (1-𝜋B)*𝜋B0*𝜋B01,
                     (1-𝜋B)*𝜋B0*(1-𝜋B01),
                     (1-𝜋B)*(1-𝜋B0)*𝜋B00,
                     (1-𝜋B)*(1-𝜋B0)*(1-𝜋B00)])
    v_B3 = np.array([1+𝜌+𝜌**2,1+𝜌+𝜌**2*v,1+𝜌*v+𝜌**2,1+𝜌*v,𝜌+𝜌**2,𝜌+𝜌**2*v,𝜌**2,0])
    v_A3 = np.array([0,0,0,0,v,v,v,v])+v_B3  
    
    phi_sign = P_B3.dot(v_B3)
    phi_nsign= P_A3.dot(v_B3)
    n_sign  = P_B3.dot(v_A3)
    n_nsign = P_A3.dot(v_A3)
    
    Uphi =    𝜌 * vn * max(phi_sign,phi_nsign)
    Un = vn + 𝜌 * vn * max(n_sign,n_nsign)
    
    return Un-Uphi


def GRBTp1q0_EXT(m=0.2,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    '''
    Note: 
    At t=2, after issuing masks, we deduct a point for people in category B.
    This should be equivalent to add a point to those in category A.
    '''

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**10) * 0.0138 * I_0
    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=0,𝜋A=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA
    
    𝜋B_coef = (1-m)*(1-d0)
    𝜋A_coef =    m *(1-d0)
    
    𝜋B = min(m/𝜋B_coef,1)
    𝜋A = (m-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0

    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=𝜋B,𝜋A=𝜋A,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(3)
    dB0, dB1, dA0, dA1 = Nation.dB0, Nation.dB1, Nation.dA0, Nation.dA1
    
    if 𝜋B<1:
        𝜋2_0_coef=        𝜋A_coef-dA
        𝜋2_1_coef=(1-𝜋B)*(𝜋B_coef-dB)
        𝜋2_2_coef=   𝜋B *(𝜋B_coef-dB)
        
        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        if 𝜋A0<1:
            𝜋3_0_coef=(1-𝜋A0)*( 𝜋2_0_coef-dA0 ) + 𝜋2_1_coef-dB0 #𝜋A00,𝜋B00
            𝜋3_1_coef=   𝜋A0 *( 𝜋2_0_coef-dA0 ) + 𝜋2_2_coef-dB1 #𝜋A01,𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
        
        elif 𝜋B0>0 and 𝜋B0<1:
            𝜋3_0_coef=               (1-𝜋B0)*(  𝜋2_1_coef-dB0 ) # 𝜋B00
            𝜋3_1_coef= 𝜋2_0_coef-dA0+   𝜋B0 *(  𝜋2_1_coef-dB0 ) + 𝜋2_2_coef-dB1  # 𝜋A01, 𝜋B10, 𝜋B01
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            
        elif 𝜋B1>0:
            𝜋3_1_coef= 𝜋2_0_coef-dA0 + 𝜋2_1_coef-dB0 + (1-𝜋B1)*( 𝜋2_2_coef-dB1 ) #𝜋A01, 𝜋B01, 𝜋B10
            𝜋3_2_coef= 𝜋B1 *( 𝜋2_2_coef-dB1 ) #𝜋B11                   
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
            

    else:
        𝜋2_0_coef=(1-𝜋A)*( 𝜋A_coef-dA )
        𝜋2_1_coef=   𝜋A *( 𝜋A_coef-dA )
        𝜋2_2_coef=         𝜋B_coef-dB

        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        if 𝜋A0<1:
            𝜋3_0_coef=(1-𝜋A0)*( 𝜋2_0_coef-dA0 ) #𝜋A00
            𝜋3_1_coef=   𝜋A0 *( 𝜋2_0_coef-dA0 ) + 𝜋2_1_coef-dA1 + 𝜋2_2_coef-dB1 #𝜋A01, 𝜋A10, 𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1) 
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            
        elif 𝜋A1>0 and 𝜋A1<1:
            𝜋3_1_coef= 𝜋2_0_coef-dA0 + (1-𝜋A1)*( 𝜋2_1_coef-dA1 ) + 𝜋2_2_coef-dB1 #𝜋A01, 𝜋A10, 𝜋B10
            𝜋3_2_coef=                    𝜋A1 *( 𝜋2_1_coef-dA1 ) #𝜋A11
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
            
        elif 𝜋B1>0:
            𝜋3_1_coef= 𝜋2_0_coef-dA0 + (1-𝜋B1)*( 𝜋2_2_coef-dB1 ) #𝜋A01, 𝜋B10
            𝜋3_2_coef= 𝜋2_1_coef-dA1 +    𝜋B1 *( 𝜋2_2_coef-dB1 ) #𝜋A11, 𝜋B11
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
    
    ICphi,ICn = IC_cons_EXT(v=v,𝜌=𝜌,πA=πA,πB=πB,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1,
                            πB00=πB00, πB01=πB01, πB10=πB10, πB11=πB11,
                            πA00=πA00, πA01=πA01, πA10=πA10, πA11=πA11)
    
    return ICphi,ICn




def pmix_func_EXT(x=1,m=0.2,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    '''
    Note: 
    At t=2, after issuing masks, we deduct a point for people in category B.
    This should be equivalent to add a point to those in category A.
    '''

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**10) * 0.0138 * I_0
    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=0,𝜋A=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA
    
    𝜋B_coef = x*(1-m)*(1-d0)
    𝜋A_coef =      m *(1-d0) + (1-x)*(1-m)*(1-d0)
    
    𝜋B = min(m/𝜋B_coef,1)
    𝜋A = (m-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0
    
    dphiB=x*dB
    dphiA=(1-x)*dB
    dnA = dA
    
    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=x*𝜋B+(1-x)*𝜋A,𝜋A=𝜋A,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(3)
    dB0, dB1, dA0, dA1 = Nation.dB0, Nation.dB1, Nation.dA0, Nation.dA1
    
    if 𝜋B<1:
        𝜋2_0_coef=         𝜋A_coef- dnA - dphiA
        𝜋2_1_coef=(1-𝜋B)*( 𝜋B_coef-dphiB )
        𝜋2_2_coef=   𝜋B *( 𝜋B_coef-dphiB )
        
        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        dphiB0 = x*(1-𝜋B) / (x*(1-𝜋B)+(1-x)*(1-𝜋A)) * dB0        
        dphiA0 = (1-x)*(1-𝜋A) / (x*(1-𝜋B)+(1-x)*(1-𝜋A)) * dB0
        dphiB1 = x*𝜋B / (x*𝜋B+(1-x)*𝜋A) * dB1
        dphiA1 = (1-x)*𝜋A / (x*𝜋B+(1-x)*𝜋A) * dB1
        
        dnA0=dA0
        dnA1=dA1
        
        if 𝜋A0<1:
            𝜋3_0_coef=(1-𝜋A0)*( 𝜋2_0_coef - dnA0 - dphiA0 ) + 𝜋2_1_coef-dphiB0 #𝜋A00,𝜋B00
            𝜋3_1_coef=   𝜋A0 *( 𝜋2_0_coef - dnA0 - dphiA0 ) + 𝜋2_2_coef-dphiB1 #𝜋A01,𝜋B10         
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
        
        elif 𝜋B0>0 and 𝜋B0<1:
            𝜋3_0_coef=                        (1-𝜋B0)*( 𝜋2_1_coef-dphiB0 ) #𝜋B00
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 +   𝜋B0 *( 𝜋2_1_coef-dphiB0 ) + 𝜋2_2_coef - dphiB1 #𝜋A01, 𝜋B01, 𝜋B10  
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            
        elif 𝜋B1>0:
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 + 𝜋2_1_coef-dphiB0 + (1-𝜋B1)*( 𝜋2_2_coef - dphiB1 ) #𝜋A01, 𝜋B01, 𝜋B10
            𝜋3_2_coef=                                               𝜋B1 *( 𝜋2_2_coef - dphiB1 ) #𝜋B11
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            

    else:
        𝜋2_0_coef=(1-𝜋A)*( 𝜋A_coef- dnA - dphiA )
        𝜋2_1_coef=   𝜋A *( 𝜋A_coef- dnA - dphiA )
        𝜋2_2_coef=         𝜋B_coef-dphiB

        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        dphiB0 = x*(1-𝜋B)/(x*(1-𝜋B)+(1-x)*(1-𝜋A))*dB0        
        dphiA0 = (1-x)*(1-𝜋A)/(x*(1-𝜋B)+(1-x)*(1-𝜋A))*dB0
        dphiB1 = x*𝜋B/(x*𝜋B+(1-x)*𝜋A)*dB1
        dphiA1 = (1-x)*𝜋A/(x*𝜋B+(1-x)*𝜋A)*dB1
        
        dnA0=dA0
        dnA1=dA1
        
        if 𝜋A0<1:
            𝜋3_0_coef= (1-𝜋A0)*( 𝜋2_0_coef-dnA0-dphiA0 ) #𝜋A00
            𝜋3_1_coef=    𝜋A0 *( 𝜋2_0_coef-dnA0-dphiA0 ) + 𝜋2_1_coef-dnA1-dphiA1 + 𝜋2_2_coef-dphiB1 #𝜋A01,𝜋A10,𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            
        elif 𝜋A1>0 and 𝜋A1<1:
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 + (1-𝜋A1)*( 𝜋2_1_coef-dnA1-dphiA1 ) + 𝜋2_2_coef-dphiB1 #𝜋A01, 𝜋A10, 𝜋B10
            𝜋3_2_coef=                            𝜋A1 *( 𝜋2_1_coef-dnA1-dphiA1 ) #𝜋A11
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
            
        elif 𝜋B1>0:
            𝜋3_1_coef=  𝜋2_0_coef - dnA0 - dphiA0 + (1-𝜋B1)*( 𝜋2_2_coef-dphiB1 ) #𝜋A01, 𝜋B10
            𝜋3_2_coef=  𝜋2_1_coef - dnA1 - dphiA1 +    𝜋B1 *( 𝜋2_2_coef-dphiB1 ) #𝜋A11, 𝜋B11                                 
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
    
    ICphi,ICn = IC_cons_EXT(v=v,𝜌=𝜌,πA=πA,πB=πB,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1,
                            πB00=πB00, πB01=πB01, πB10=πB10, πB11=πB11,
                            πA00=πA00, πA01=πA01, πA10=πA10, πA11=πA11)
    
    return ICphi

def qmix_func_EXT(x=0,m=0.2,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    '''
    Note: 
    At t=2, after issuing masks, we deduct a point for people in category B.
    This should be equivalent to add a point to those in category A.
    '''

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**10) * 0.0138 * I_0
    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=0,𝜋A=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA
    
    𝜋B_coef = (1-m)*(1-d0) +   x *m*(1-d0)
    𝜋A_coef =               (1-x)*m*(1-d0)
    
    𝜋B = min(m/𝜋B_coef,1)
    𝜋A = (m-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0

    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=𝜋B,𝜋A=𝜋A,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(3)
    dB0, dB1, dA0, dA1 = Nation.dB0, Nation.dB1, Nation.dA0, Nation.dA1
    
    if 𝜋B<1:
        𝜋2_0_coef=                         (1-x)*(m*(1-d0)-dA)
        𝜋2_1_coef=(1-𝜋B)*( (1-m)*(1-d0)-dB +  x *(m*(1-d0)-dA) )
        𝜋2_2_coef=   𝜋B *( (1-m)*(1-d0)-dB +  x *(m*(1-d0)-dA) )
        
        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        dBn0 = x*(1-𝜋B)/(x*(1-𝜋B)+(1-x)*(1-𝜋A))*dA0        
        dAn0 = (1-x)*(1-𝜋A)/(x*(1-𝜋B)+(1-x)*(1-𝜋A))*dA0
        dBn1 = x*𝜋B/(x*𝜋B+(1-x)*𝜋A)*dA1
        dAn1 = (1-x)*𝜋A/(x*𝜋B+(1-x)*𝜋A)*dA1
        
        if 𝜋A0<1:
            𝜋3_0_coef= (1-𝜋A0)*(1-x)*(m*(1-d0)-dA - dAn0) #𝜋A00
            𝜋3_1_coef=(   𝜋A0 *(1-x)*(m*(1-d0)-dA - dAn0) + 
                       (1-𝜋B)*((1-m)*(1-d0)-dB)-dB0 + (1-𝜋B)*x*(m*(1-d0)-dA)-dBn0 ) #𝜋A01, 𝜋B00
            𝜋3_2_coef=    𝜋B *((1-m)*(1-d0)-dB)-dB1 +    𝜋B *x*(m*(1-d0)-dA)-dBn1 # 𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B00=𝜋A01
            𝜋B10=(m-𝜋3_0_coef-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋B01=𝜋A11=𝜋B10
            𝜋B11=0
        
        elif 𝜋B0>0 and 𝜋B0<1:
            𝜋3_1_coef=((1-x)*(m*(1-d0)-dA) - dAn0 + 
                       (1-𝜋B0)*( (1-𝜋B)*((1-m)*(1-d0)-dB)-dB0 + (1-𝜋B)*x*(m*(1-d0)-dA)-dBn0 )  ) # 𝜋A01, 𝜋B00
            𝜋3_2_coef=(   𝜋B0 *( (1-𝜋B)*((1-m)*(1-d0)-dB)-dB0 + (1-𝜋B)*x*(m*(1-d0)-dA)-dBn0 ) + 
                                    𝜋B *((1-m)*(1-d0)-dB)-dB1 +    𝜋B *x*(m*(1-d0)-dA)-dBn1) # 𝜋B01, 𝜋B10
            
            𝜋A00=1
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B00=𝜋A01
            𝜋B10=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋B01=𝜋A11=𝜋B10
            𝜋B11=0
            
        elif 𝜋B1>0:
            𝜋3_1_coef= (1-x)*(m*(1-d0)-dA) - dAn0 #𝜋A01
            𝜋3_2_coef=(         (1-𝜋B)*((1-m)*(1-d0)-dB)-dB0 + (1-𝜋B)*x*(m*(1-d0)-dA)-dBn0 + 
                       (1-𝜋B1)*(   𝜋B *((1-m)*(1-d0)-dB)-dB1 +    𝜋B *x*(m*(1-d0)-dA)-dBn1 )  ) #𝜋B01, 𝜋B10
            𝜋3_3_coef=    𝜋B1 *(   𝜋B *((1-m)*(1-d0)-dB)-dB1 +    𝜋B *x*(m*(1-d0)-dA)-dBn1 )  #𝜋B11
            
            𝜋A00=1
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B00=𝜋A01
            𝜋B10=min((m-𝜋3_1_coef)/𝜋3_2_coef,1) if 𝜋A01==1 else 0
            𝜋B01=𝜋A11=𝜋B10
            𝜋B11=(m-𝜋3_1_coef-𝜋3_2_coef)/𝜋3_3_coef if 𝜋B10==1 else 0
            

    else:
        𝜋2_0_coef=(1-𝜋A)*(1-x)*(m*(1-d0)-dA)
        𝜋2_1_coef=   𝜋A *(1-x)*(m*(1-d0)-dA)
        𝜋2_2_coef= (1-m)*(1-d0)-dB + x*(m*(1-d0)-dA)

        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        dBn0 = x*(1-𝜋B)/(x*(1-𝜋B)+(1-x)*(1-𝜋A))*dA0        
        dAn0 = (1-x)*(1-𝜋A)/(x*(1-𝜋B)+(1-x)*(1-𝜋A))*dA0
        dBn1 = x*𝜋B/(x*𝜋B+(1-x)*𝜋A)*dA1
        dAn1 = (1-x)*𝜋A/(x*𝜋B+(1-x)*𝜋A)*dA1
        
        if 𝜋A0<1:
            𝜋3_0_coef=(1-𝜋A0)*( (1-𝜋A)*(1-x)*(m*(1-d0)-dA)-dAn0 ) # 𝜋A00
            𝜋3_1_coef=   𝜋A0 *( (1-𝜋A)*(1-x)*(m*(1-d0)-dA)-dAn0 )+ 𝜋A*(1-x)*(m*(1-d0)-dA)-dAn1 # 𝜋A01, 𝜋A10
            𝜋3_2_coef= (1-m)*(1-d0)-dB-dB1 + x*(m*(1-d0)-dA)-dBn1 # 𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋A01=min((m-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
            𝜋A10=𝜋B00=𝜋A01
            𝜋B10=(m-𝜋3_0_coef-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋B01=𝜋A11=𝜋B10
            𝜋B11=0
            
        elif 𝜋A1>0 and 𝜋A1<1:
            𝜋3_1_coef= (1-𝜋A)*(1-x)*(m*(1-d0)-dA)-dAn0 + (1-𝜋A1)*( 𝜋A*(1-x)*(m*(1-d0)-dA)-dAn1 ) #𝜋A01, 𝜋A10
            𝜋3_2_coef= 𝜋A1*( 𝜋A*(1-x)*(m*(1-d0)-dA)-dAn1 ) + (1-m)*(1-d0)-dB-dB1 + x*(m*(1-d0)-dA)-dBn1 # 𝜋A11, 𝜋B10
            
            𝜋A00=1
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B00=𝜋A01
            𝜋B10=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋B01=𝜋A11=𝜋B10
            𝜋B11=0
            
        elif 𝜋B1>0:
            𝜋3_1_coef= (1-𝜋A)*(1-x)*(m*(1-d0)-dA)-dAn0 # 𝜋A01
            𝜋3_2_coef=(   𝜋A *(1-x)*(m*(1-d0)-dA)-dAn1 + 
                       (1-𝜋B1)*( (1-m)*(1-d0)-dB-dB1 + x*(m*(1-d0)-dA)-dBn1 )) # 𝜋A11, 𝜋B10
            𝜋3_3_coef=    𝜋B1 *( (1-m)*(1-d0)-dB-dB1 + x*(m*(1-d0)-dA)-dBn1) #𝜋B11
            
            𝜋A00=1
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B00=𝜋A01
            𝜋B10=min((m-𝜋3_1_coef)/𝜋3_2_coef,1) if 𝜋A01==1 else 0
            𝜋B01=𝜋A11=𝜋B10
            𝜋B11=(m-𝜋3_1_coef-𝜋3_2_coef)/𝜋3_3_coef if 𝜋B10==1 else 0
    
    ICphi,ICn = IC_cons_EXT(v=v,𝜌=𝜌,πA=πA,πB=πB,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1,
                            πB00=πB00, πB01=πB01, πB10=πB10, πB11=πB11,
                            πA00=πA00, πA01=πA01, πA10=πA10, πA11=πA11)
    
    return ICn


def Mix_computeProb_EXT(x=1,y=0,m=0.2,I_0=0.05,σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    '''
    Note: 
    At t=2, after issuing masks, we deduct a point for people in category B.
    This should be equivalent to add a point to those in category A.
    '''
    
    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**10) * 0.0138 * I_0
    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=0,𝜋A=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA
    
    dphiB=   x *dB
    dphiA=(1-x)*dB
    dnB  =   y *dA
    dnA  =(1-y)*dA
    
    𝜋B_coef =    x *(1-m)*(1-d0) +    y *m*(1-d0)
    𝜋A_coef = (1-x)*(1-m)*(1-d0) + (1-y)*m*(1-d0)
    
    𝜋B = min(m/𝜋B_coef,1)
    𝜋A = (m-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0

    Nation = T4_SIRD(S=1-I_0,I=I_0,π0=m,𝜋B=x*𝜋B+(1-x)*𝜋A,𝜋A=y*𝜋B+(1-y)*𝜋A,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(3)
    dB0, dB1, dA0, dA1 = Nation.dB0, Nation.dB1, Nation.dA0, Nation.dA1
    
    dphiB0 = x*(1-𝜋B) / (x*(1-𝜋B)+(1-x)*(1-𝜋A)) * dB0      if x*(1-𝜋B)!=0 else 0   
    dphiA0 = (1-x)*(1-𝜋A) / (x*(1-𝜋B)+(1-x)*(1-𝜋A)) * dB0  if (1-x)*(1-𝜋A)!=0 else 0
    dphiB1 = x*𝜋B / (x*𝜋B+(1-x)*𝜋A) * dB1                  if x*𝜋B!=0 else 0
    dphiA1 = (1-x)*𝜋A / (x*𝜋B+(1-x)*𝜋A) * dB1              if (1-x)*𝜋A!=0 else 0

    dnB0 = y*(1-𝜋B) / (y*(1-𝜋B)+(1-y)*(1-𝜋A)) * dA0      if y*(1-𝜋B)!=0 else 0    
    dnA0 = (1-y)*(1-𝜋A) / (y*(1-𝜋B)+(1-y)*(1-𝜋A)) * dA0  if (1-y)*(1-𝜋A)!=0 else 0
    dnB1 = y*𝜋B / (y*𝜋B+(1-y)*𝜋A) * dA1                  if y*𝜋B!=0 else 0
    dnA1 = (1-y)*𝜋A/(y*𝜋B+(1-y)*𝜋A) * dA1                if (1-y)*𝜋A!=0 else 0
    
    if 𝜋B<1:
        𝜋2_0_coef=          𝜋A_coef - dnA - dphiA
        𝜋2_1_coef=(1-𝜋B)*(  𝜋B_coef - dnB - dphiB )
        𝜋2_2_coef=   𝜋B *(  𝜋B_coef - dnB - dphiB )
        
        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        if 𝜋A0<1:
            𝜋3_0_coef= (1-𝜋A0)*( 𝜋2_0_coef-dnA0-dphiA0 ) + 𝜋2_1_coef-dnB0-dphiB0  #𝜋A00, 𝜋B00
            𝜋3_1_coef=    𝜋A0 *( 𝜋2_0_coef-dnA0-dphiA0 ) + 𝜋2_2_coef-dnB1-dphiB1  #𝜋A01, 𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=(m-𝜋3_0_coef)/𝜋3_1_coef if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
        
        elif 𝜋B0>0 and 𝜋B0<1:
            𝜋3_0_coef=                        (1-𝜋B0)*( 𝜋2_1_coef-dnB0-dphiB0 ) #𝜋B00
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 +   𝜋B0 *( 𝜋2_1_coef-dnB0-dphiB0 ) + 𝜋2_2_coef-dnB1-dphiB1 #𝜋A01, 𝜋B01, 𝜋B10                         
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=(m-𝜋3_0_coef)/𝜋3_1_coef if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            
        elif 𝜋B1>0:
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 + 𝜋2_1_coef-dnB0-dphiB0 + (1-𝜋B1)*( 𝜋2_2_coef-dnB1-dphiB1 )#𝜋A01, 𝜋B01, 𝜋B10
            𝜋3_2_coef=                                                    𝜋B1 *( 𝜋2_2_coef-dnB1-dphiB1 )#𝜋B11                       
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1) 
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
            

    else:
        𝜋2_0_coef=(1-𝜋A)*( 𝜋A_coef - dnA - dphiA )
        𝜋2_1_coef=   𝜋A *( 𝜋A_coef - dnA - dphiA )
        𝜋2_2_coef=         𝜋B_coef - dnB - dphiB

        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0
        
        if 𝜋A0<1:
            𝜋3_0_coef= (1-𝜋A0)*( 𝜋2_0_coef-dnA0-dphiA0 ) #𝜋A00
            𝜋3_1_coef=    𝜋A0 *( 𝜋2_0_coef-dnA0-dphiA0 ) + 𝜋2_1_coef-dnA1-dphiA1 + 𝜋2_2_coef-dnB1-dphiB1 #𝜋A01, #𝜋A10, 𝜋B10
            
            𝜋A00=min(m/𝜋3_0_coef,1)
            𝜋B00=𝜋A00
            𝜋A01=(m-𝜋3_0_coef)/𝜋3_1_coef if 𝜋A00==1 else 0
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=0
            𝜋A11=𝜋B11
            
        elif 𝜋A1>0 and 𝜋A1<1:
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 + (1-𝜋A1)*( 𝜋2_1_coef-dnA1-dphiA1 ) #𝜋A01, #𝜋A10
            𝜋3_2_coef=                            𝜋A1 *( 𝜋2_1_coef-dnA1-dphiA1 ) + 𝜋2_2_coef-dnB1-dphiB1  #𝜋A11, #𝜋B10
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
            
        elif 𝜋B1>0:
            𝜋3_1_coef= 𝜋2_0_coef-dnA0-dphiA0 + 𝜋2_1_coef-dnA1-dphiA1 + (1-𝜋B1)*( 𝜋2_2_coef-dnB1-dphiB1 )  #𝜋A01, 𝜋A10, 𝜋B10
            𝜋3_2_coef= 𝜋B1 *( 𝜋2_2_coef-dnB1-dphiB1 ) #𝜋B11 
           
            
            𝜋A00=1
            𝜋B00=𝜋A00
            𝜋A01=min(m/𝜋3_1_coef,1)
            𝜋A10=𝜋B10=𝜋B01=𝜋A01
            𝜋B11=(m-𝜋3_1_coef)/𝜋3_2_coef if 𝜋A01==1 else 0
            𝜋A11=𝜋B11
                        
    return πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11
                        
                        
                        

def GRBT_EXT(m=0.2,I_0=0.01,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,vo=0.5,vn=0.7):
    
    ICphi_sep, ICn_sep = GRBTp1q0_EXT(m=m,I_0=I_0,v=vo/vn,𝜌=𝜌,σo=σo,σn=σn,δo=δo,δn=δn)
    
    #### Fully-Separating Equilibrium ####
    if ICphi_sep>=0 and ICn_sep<=0:
        πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11 = Mix_computeProb_EXT(1,0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
        D_val = GRBT_evalD_EXT(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                               πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
        S4,I4 = GRBT_evalT4_EXT(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=4)
        Ud = Udiff_EXT(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
        
        return m,1,0,ICphi_sep,ICn_sep,πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11,S4,I4,D_val,Ud
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    if ICphi_sep<0:
        #### Partial-Separating Equilibrium. #### 
        # People without mask play mix strategy
        p_res=root(pmix_func_EXT,0.75,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n),method='excitingmixing',tol=10e-14)

        if p_res.success and p_res.x>0 and p_res.x<1:
            p_star=p_res.x
            πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11 = Mix_computeProb_EXT(x=p_star,y=0,
                                                                                                m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
            D_val = GRBT_evalD_EXT(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                   πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
            S4,I4 = GRBT_evalT4_EXT(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                    πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                    πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                    S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=4)
            ICphi, ICn = IC_cons_EXT(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
            Ud = Udiff_EXT(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
            
            return m,p_star,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11,S4,I4,D_val,Ud
        
        #### Pooling Equlibrium ####
        # No people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11 = Mix_computeProb_EXT(x=0,y=0,
                                                                                            m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
        D_val = GRBT_evalD_EXT(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                               πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
        S4,I4 = GRBT_evalT4_EXT(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=4)
        ICphi, ICn = IC_cons_EXT(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                 πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                 πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
        Ud = Udiff_EXT(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)

        return m,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11,S4,I4,D_val,Ud

        
        
    if ICn_sep>0:
        #### Partial-Separating Equilibrium. #### 
        # People having mask play mix strategy
        q_res=root(qmix_func_EXT,0.2,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n),method='broyden1',tol=10e-12)

        if q_res.success and q_res.x>0 and q_res.x<1:
            q_star=q_res.x
            πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11 = Mix_computeProb_EXT(x=1,y=q_star,
                                                                                                m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
            D_val = GRBT_evalD_EXT(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                   πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
            S4,I4 = GRBT_evalT4_EXT(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                    πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                    πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                    S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=4)
            ICphi, ICn = IC_cons_EXT(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
            Ud = Udiff_EXT(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)

            return m,1,q_star,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11,S4,I4,D_val,Ud 
        
        #### Pooling Equlibrium ####
        # All people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11 = Mix_computeProb_EXT(x=1,y=1,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
        D_val = GRBT_evalD_EXT(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                               πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
        S4,I4 = GRBT_evalT4_EXT(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=4)
        ICphi, ICn = IC_cons_EXT(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                 πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                 πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
        Ud = Udiff_EXT(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                     πB00=πB00,πB01=πB01,πB10=πB10,πB11=πB11,
                                     πA00=πA00,πA01=πA01,πA10=πA10,πA11=πA11)
        
        return m,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,πA00,πB00,πA01,πA10,πB01,πB10,πA11,πB11,S4,I4,D_val,Ud
    
    
    
def SRA1_EXT(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**10) * 0.0138 * I_0
    Nation = T4_SIRD(S=1-I_0,I=I_0,𝜋0=m,𝜋B=0,𝜋A=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    𝜋1 = m/(1-d0)
    
    Nation = T4_SIRD(S=1-I_0,I=I_0,𝜋0=m,𝜋B=𝜋1,𝜋A=𝜋1,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(3)
    dB0,dB1,dA0,dA1 = Nation.dB0, Nation.dB1, Nation.dA0, Nation.dA1

    𝜋2_coef = (1-m)*(1-d0)-dB + m*(1-d0)-dA
    𝜋2 = m/𝜋2_coef
    
    𝜋3_coef = (1-𝜋1)*((1-m)*(1-d0)-dB)-dB0 + 𝜋1*((1-m)*(1-d0)-dB)-dB1 + (1-𝜋1)*(m*(1-d0)-dA)-dA0 + 𝜋1*(m*(1-d0)-dA)-dA1
    𝜋3 = m/𝜋3_coef

    func = evalD_EXT(S=1-I_0,I=I_0,π0=m,𝜋B=𝜋1,𝜋A=𝜋1,𝜋B0=𝜋2,𝜋A0=𝜋2,𝜋B1=𝜋2,𝜋A1=𝜋2,
                         𝜋B00=𝜋3,𝜋B01=𝜋3,𝜋B10=𝜋3,𝜋B11=𝜋3,
                         𝜋A00=𝜋3,𝜋A01=𝜋3,𝜋A10=𝜋3,𝜋A11=𝜋3,                         
                         σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    S4,I4 = evalT4_EXT(S=1-I_0,I=I_0,π0=m,𝜋B=𝜋1,𝜋A=𝜋1,𝜋B0=𝜋2,𝜋A0=𝜋2,𝜋B1=𝜋2,𝜋A1=𝜋2,
                           𝜋B00=𝜋3,𝜋B01=𝜋3,𝜋B10=𝜋3,𝜋B11=𝜋3,
                           𝜋A00=𝜋3,𝜋A01=𝜋3,𝜋A10=𝜋3,𝜋A11=𝜋3,
                           σo=σo,σn=σn,δo=δo,δn=δn,t=3)
    Ud = Udiff_EXT(vo=0.5,vn=0.7,𝜌=1,𝜋B=𝜋1,𝜋A=𝜋1,𝜋B0=𝜋2,𝜋A0=𝜋2,𝜋B1=𝜋2,𝜋A1=𝜋2,
                           𝜋B00=𝜋3,𝜋B01=𝜋3,𝜋B10=𝜋3,𝜋B11=𝜋3,
                           𝜋A00=𝜋3,𝜋A01=𝜋3,𝜋A10=𝜋3,𝜋A11=𝜋3)

    return {'func':func,'𝜋1':𝜋1,'𝜋2':𝜋2,'𝜋3':𝜋3,'S4':S4,'I4':I4,'Ud':Ud}

def SRA2_EXT(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**10) * 0.0138 * I_0
    Nation = T4_SIRD(S=1-I_0,I=I_0,𝜋0=m,𝜋B=0,𝜋A=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA

    𝜋1 = m/(1-d0)
    
    Nation = T4_SIRD(S=1-I_0,I=I_0,𝜋0=m,𝜋B=𝜋1,𝜋A=𝜋1,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(3)
    dB0,dB1,dA0,dA1 = Nation.dB0, Nation.dB1, Nation.dA0, Nation.dA1

    𝜋20_coef = (1-𝜋1)*((1-m)*(1-d0)-dB + m*(1-d0)-dA)
    𝜋21_coef =    𝜋1 *((1-m)*(1-d0)-dB + m*(1-d0)-dA)
    
    𝜋20 = min(m/𝜋20_coef,1)
    𝜋21 = (m-𝜋20_coef)/𝜋21_coef if 𝜋20==1 else 0
    
    if 𝜋20 <1:
        𝜋30_coef = (1-𝜋20)*( 𝜋20_coef-dB0-dA0 )
        𝜋31_coef =    𝜋20 *( 𝜋20_coef-dB0-dA0 ) + 𝜋21_coef-dB1-dA1
        
        𝜋30 = min(m/𝜋30_coef,1)
        𝜋31 = (m-𝜋30_coef)/𝜋31_coef if 𝜋30==1 else 0
        𝜋32 = 0
        
        
    else:
        𝜋31_coef = 𝜋20_coef-dB0-dA0 + (1-𝜋21)*( 𝜋21_coef-dB1-dA1 )
        𝜋32_coef =                       𝜋21 *( 𝜋21_coef-dB1-dA1 )    
    
        𝜋30 = 1
        𝜋31 = min(m/𝜋31_coef,1)
        𝜋32 = (m-𝜋31_coef)/𝜋32_coef if 𝜋31==1 else 0
        
    func = evalD_EXT(S=1-I_0,I=I_0,π0=m,𝜋B=𝜋1,𝜋A=𝜋1,𝜋B0=𝜋20,𝜋A0=𝜋20,𝜋B1=𝜋21,𝜋A1=𝜋21,
                     𝜋B00=𝜋30,𝜋B01=𝜋31,𝜋B10=𝜋31,𝜋B11=𝜋32,
                     𝜋A00=𝜋30,𝜋A01=𝜋31,𝜋A10=𝜋31,𝜋A11=𝜋32,                         
                     σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    S4,I4 = evalT4_EXT(S=1-I_0,I=I_0,π0=m,𝜋B=𝜋1,𝜋A=𝜋1,𝜋B0=𝜋20,𝜋A0=𝜋20,𝜋B1=𝜋21,𝜋A1=𝜋21,
                       𝜋B00=𝜋30,𝜋B01=𝜋31,𝜋B10=𝜋31,𝜋B11=𝜋32,
                       𝜋A00=𝜋30,𝜋A01=𝜋31,𝜋A10=𝜋31,𝜋A11=𝜋32,   
                       σo=σo,σn=σn,δo=δo,δn=δn,t=3)
    Ud = Udiff_EXT(vo=0.5,vn=0.7,𝜌=1,𝜋B=𝜋1,𝜋A=𝜋1,𝜋B0=𝜋20,𝜋A0=𝜋20,𝜋B1=𝜋21,𝜋A1=𝜋21,
                       𝜋B00=𝜋30,𝜋B01=𝜋31,𝜋B10=𝜋31,𝜋B11=𝜋32,
                       𝜋A00=𝜋30,𝜋A01=𝜋31,𝜋A10=𝜋31,𝜋A11=𝜋32)

    return {'func':func,'𝜋1':𝜋1,'𝜋20':𝜋20,'𝜋21':𝜋21,'𝜋30':𝜋30,'𝜋31':𝜋31,'𝜋32':𝜋32,'S4':S4,'I4':I4,'Ud':Ud}