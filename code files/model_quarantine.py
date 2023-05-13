import numpy as np
import pandas as pd
from scipy.optimize import root

'''
Qua_SIRD: describe the dynamic of SIRD and calculate number of death at period t=1 for resource constraints.
evalD_Qua: calculate number of deaths for SRA-I and II
evalT3_Qua: calculate state variables S and I under SRA-I and II mechanisms at period t=T
GRBT_evalD_Qua: calculate number of deaths for GRBT
GRBT_evalT3_Qua: calculate state variables S and I under GRBT mechanisms at period t=T
IC_cons: calculate gain from early participation for both groups of people
Udiff: calculate utility difference between people with and without mask initially

GRBTp1q0_Qua: return gain from early participation under fully separate senerio 
Mix_computeProb_Qua: given the shares of both groups of people who early participated in GRBT, compute the probs of receiving masks 
pmix_func_Qua: given the share of people without mask who early participated in GRBT, return their gain from early participation
qmix_func_Qua: given the share of people with mask who early participated in GRBT, return their gain from early participation
GRBT_Qua: given initial coverage rate m0, determine types of equilibrium and return number of deaths, probs of receiving masks, with other information
SRA1_Qua: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-I mechanism
SRA2_Qua: given initial coverage rate m0, return return number of deaths and probs of receiving masks under SRA-II mechanism
'''

class Qua_SIRD:
    
    def __init__(self,S=0.9,        # initial susceptible
                      I=0.1,        # initial infectious
                      R=0,          # initial recovered
                      D=0,          # initial died
                 
                      I_I=0,        # initial isolated infectious
                      R_I=0,        # initial isolated recovered
                 
                      𝛽=2.4/(18/14),# basic transmission rate. R0=2.4 and it takes 18 days to leave I state in average.
                                    # Furthermore, a time unit is 14 days here.
                      𝛾=1-(17/18)**14,# propotion of people that will leave state I is one minus those does not leave in fourteen days 
                      𝛼=0.0138,     # propotion that will die after leave state I.
                      𝜃=0.2,        # quarantine power
                     
                      𝛾_I=1-(17/18)**14,
                      𝛼_I=2*0.0138,
                                     
                      
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
        self.R, self.D, self.I_I, self.R_I  = R, D, I_I, R_I
        self.𝛽, self.𝛾, self.𝜃, self.𝛼, self.𝛾_I, self.𝛼_I = 𝛽, 𝛾, 𝛼, 𝜃, 𝛾_I, 𝛼_I
        self.σo, self.σn, self.δo, self.δn = σo, σn, δo, δn

        self.T, self.𝜋0 = T, 𝜋0
        self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1 = πB, πA, πB0, πA0, πB1, πA1
        
    def evaluate_change(self):
        T = self.T
        𝛽, 𝛾, 𝛼, 𝜃, 𝛾_I, 𝛼_I = self.𝛽, self.𝛾, self.𝜃, self.𝛼, self.𝛾_I, self.𝛼_I
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        𝜋0, πB, πA, πB0, πA0, πB1, πA1 = self.𝜋0, self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1

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
            ##### compute number of death for resource constraints.
            self.kB, self.kA = ( 𝛾*𝛼 + 𝜃*(1-𝛾) ) * self.I[0], ( 𝛾*𝛼 + 𝜃*(1-𝛾) ) * self.I[1]
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
            transition_2 = np.array([[1-𝜋B0,0,0,0,0,1-𝜋A0,    0,    0],
                                     [    0,0,0,0,0,    0,1-𝜋B1,1-𝜋A1],
                                     [  𝜋B0,0,0,0,0,  𝜋A0,  𝜋B1,  𝜋A1]])
            S_mask = transition_2.dot(self.S) # 3x1
            I_mask = transition_2.dot(self.I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T>=3:
            transition = np.array([[0,0,0],
                                   [0,0,0],
                                   [1,1,1]])
            S_mask = transition.dot(self.S) # 3x1
            I_mask = transition.dot(self.I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        # I's dynamics
        dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * self.I_I
        dR_I = 𝛾_I * (1-𝛼_I) * self.I_I
        dR = 𝛾 * (1-𝛼) * I_mask
        dD = 𝛾 * 𝛼 * I_mask 
        dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 
        

        nS = S_mask - dS
        nI = I_mask + dI
        nI_I = self.I_I + dI_I
        nR_I = self.R_I + dR_I
        nR = self.R + sum(dR)
        nD = self.D + sum(dD) + 𝛾_I * 𝛼_I * self.I_I
        

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
        return(np.array([nS,nI,nI_I,nR_I,nR,nD]))
    
    def update(self):
        
        change = self.evaluate_change()
        self.S, self.I, self.I_I, self.R_I, self.R, self.D = change
        self.T = self.T+1

    def severalupdates(self,t):
        for _ in range(t):
            self.update()
        return(self.D)
    

def evalD_Qua(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   S=0.9,I=0.1,I_I=0,R_I=0,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,𝛾_I=1-(17/18)**14,𝛼_I=2*0.0138,𝜃=0.2,
                   T=0,t=10,π0=0.2,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    I_I, R_I, R, D = I_I, R_I, R, D
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 


            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

            
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

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 


            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

            
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

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        elif T==2:
            
            transition_2 = np.array([[1-πB0,0,0,0,0,1-πA0,    0,    0],
                                     [    0,0,0,0,0,    0,1-πB1,1-πA1],
                                     [  πB0,0,0,0,0,  πA0,  πB1,  πA1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I
            
            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        elif T>=3:
            
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 


            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        T=T+1
    
    return(D)



def GRBT_evalD_Qua(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   S=0.9,I=0.1,I_I=0,R_I=0,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,𝛾_I=1-(17/18)**14,𝛼_I=2*0.0138,𝜃=0.2,
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 


            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I
            
            
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

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 


            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

            
            transition_mask = np.eye(8)

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        elif T==2:
            
            transition_2 = np.array([[1-πA0,    0,1-πA0,    0,1-πB0,    0,1-πB0,    0],
                                     [    0,1-πA1,    0,1-πA1,    0,1-πB1,    0,1-πB1],
                                     [  πA0,  πA1,  πA0,  πA1,  πB0,  πB1,  πB0,  πB1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 


            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I
            
            
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        elif T>=3:
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
         
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        T=T+1
    
    return(D)

def evalT3_Qua(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
           S=0.9,I=0.1,I_I=0,R_I=0,R=0,D=0,
           𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,𝛾_I=1-(17/18)**14,𝛼_I=2*0.0138,𝜃=0.2,
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I


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

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I
            
            
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

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        elif T==2:

            transition_2 = np.array([[1-πB0,0,0,0,0,1-πA0,    0,    0],
                                     [    0,0,0,0,0,    0,1-πB1,1-πA1],
                                     [  πB0,0,0,0,0,  πA0,  πB1,  πA1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1


            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I
            

            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])


            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD
            
        elif T>=3:

            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix

            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I


            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        T=T+1
    
    return sum(S),sum(I)

def GRBT_evalT3_Qua(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   S=0.9,I=0.1,I_I=0,R_I=0,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,𝛾_I=1-(17/18)**14,𝛼_I=2*0.0138,𝜃=0.2,
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

            
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

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        
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
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I

           
            transition_mask = np.eye(8)

            
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        elif T==2:
           
            transition_2 = np.array([[1-πA0,    0,1-πA0,    0,1-πB0,    0,1-πB0,    0],
                                     [    0,1-πA1,    0,1-πA1,    0,1-πB1,    0,1-πB1],
                                     [  πA0,  πA1,  πA0,  πA1,  πB0,  πB1,  πB0,  πB1]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

          
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
     
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I's dynamics
            dI_I = sum(𝜃 * (1-𝛾) * I_mask) - 𝛾_I * I_I
            dR_I = 𝛾_I * (1-𝛼_I) * I_I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 
            dI = dS - dR - dD - 𝜃 * (1-𝛾) * I_mask 

            nS = S_mask - dS
            nI = I_mask + dI
            nI_I = I_I + dI_I
            nR_I = R_I + dR_I
            nR = R + sum(dR)
            nD = D + sum(dD) + 𝛾_I * 𝛼_I * I_I
            
          
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

  
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,I_I,R_I,R,D = nS,nI,nI_I,nR_I,nR,nD

        T=T+1
    
    return sum(S),sum(I)





def IC_cons(v=0.5,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    phi_sign = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌
    phi_nsign= 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*𝜋A0*𝜌
    n_sign   = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*(v+𝜋B0*𝜌)
    n_nsign  = 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌)

    ICphi = phi_sign-phi_nsign
    ICn   = n_sign-n_nsign
    return ICphi, ICn

def Udiff(vo=0.5,vn=0.7,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    v = vo/vn
    phi_sign = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌
    phi_nsign= 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*𝜋A0*𝜌
    n_sign   = 𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*(v+𝜋B0*𝜌)
    n_nsign  = 𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌)
    
    Uphi =    𝜌 * vn * max(phi_sign,phi_nsign)
    Un = vn + 𝜌 * vn * max(n_sign,n_nsign)
    
    return Uphi,Un,Un-Uphi

def GRBTp1q0_Qua(m=0.2,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    k0 = ((1-(17/18)**14) * 0.0138 + 0.2*(1-(1-(17/18)**14))) * I_0
    Nation = Qua_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    kB, kA = Nation.kB, Nation.kA

    𝜋B_coef = (1-m)*(1-k0)
    𝜋A_coef =    m *(1-k0)
    𝜋B = min(m/𝜋B_coef,1)
    𝜋A = (m-𝜋B_coef)/𝜋A_coef if 𝜋B==1 else 0

    
    if 𝜋B<1:
        𝜋2_0_coef=           m *(1-k0)-kA
        𝜋2_1_coef=(1-𝜋B)*((1-m)*(1-k0)-kB)
        𝜋2_2_coef=   𝜋B *((1-m)*(1-k0)-kB)
        
        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋A0==1 else 0
        𝜋A1 = 1 if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0

    else:
        𝜋2_0_coef=(1-𝜋A)*(m*(1-k0)-kA)
        𝜋2_1_coef=   𝜋A *(m*(1-k0)-kA)
        𝜋2_2_coef=       (1-m)*(1-k0)-kB

        𝜋A0 = min(m/𝜋2_0_coef,1)
        𝜋B0 = 1 if 𝜋A0==1 else 0
        𝜋A1 = min((m-𝜋2_0_coef)/𝜋2_1_coef,1) if 𝜋B0==1 else 0
        𝜋B1 = (m-𝜋2_0_coef-𝜋2_1_coef)/𝜋2_2_coef if 𝜋A1==1 else 0

    ICphi, ICn = IC_cons(v=v,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
    
    return ICphi,ICn

def Mix_computeProb_Qua(x,y,m,I_0=0.05,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    k0 = ((1-(17/18)**14) * 0.0138 + 0.2*(1-(1-(17/18)**14))) * I_0
    Nation = Qua_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    kB, kA = Nation.kB, Nation.kA

    # t=1 
    πB_coef=   x *(1-m)*(1-k0) +    y *m*(1-k0)
    πA_coef=(1-x)*(1-m)*(1-k0) + (1-y)*m*(1-k0)
    πB = min(m/πB_coef,1) 
    πA = (m-πB_coef)/πA_coef if πB==1 else 0

    # t=2 
    if πB<1:
        π2_0_coef=        (1-x)*((1-m)*(1-k0)-kB) + (1-y)*(m*(1-k0)-kA)
        π2_1_coef=(1-πB)*(   x *((1-m)*(1-k0)-kB) +    y *(m*(1-k0)-kA) )
        π2_2_coef=   πB *(   x *((1-m)*(1-k0)-kB) +    y *(m*(1-k0)-kA) )

        πA0 = min(m/π2_0_coef,1)
        πB0 = min((m-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*((1-x)*((1-m)*(1-k0)-kB) + (1-y)*(m*(1-k0)-kA) )
        π2_1_coef=   πA *((1-x)*((1-m)*(1-k0)-kB) + (1-y)*(m*(1-k0)-kA) )
        π2_2_coef=           x *((1-m)*(1-k0)-kB) +    y *(m*(1-k0)-kA)
        

        πA0 = min(m/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0
    
    return πB,πA,πA0,πB0,πA1,πB1

def pmix_func_Qua(x,m,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    k0 = ((1-(17/18)**14) * 0.0138 + 0.2*(1-(1-(17/18)**14))) * I_0
    Nation = Qua_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    kB, kA = Nation.kB, Nation.kA

    # t=1 πB for those who sign up， πA for those who don't. We prioritize those who sign up over those who don't
    πB_coef=x*(1-k0)*(1-m)
    πA_coef=(1-x)*(1-k0)*(1-m)+(1-k0)*m
    πB = min(m/πB_coef,1)
    πA = (m-πB_coef)/πA_coef if πB==1 else 0

    # two cases at t=2
    if πB<1:
        π2_0_coef=        (1-x)*((1-m)*(1-k0)-kB) + m*(1-k0)-kA
        π2_1_coef= (1-πB)*   x *((1-m)*(1-k0)-kB)
        π2_2_coef=    πB *   x *((1-m)*(1-k0)-kB)


        πA0 = min(m/π2_0_coef,1)
        πB0 = min((m-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*( (1-x)*((1-m)*(1-k0)-kB) + m*(1-k0)-kA )
        π2_1_coef=   πA *( (1-x)*((1-m)*(1-k0)-kB) + m*(1-k0)-kA )
        π2_2_coef= x *((1-m)*(1-k0)-kB)

        πA0 = min(m/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    phi_sign = πB*(1+𝜌*(πB1+(1-πB1)*v)) + (1-πB)*πB0*𝜌
    phi_nsign= πA*(1+𝜌*(πA1+(1-πA1)*v)) + (1-πA)*πA0*𝜌

    return phi_sign-phi_nsign

def qmix_func_Qua(x,m,I_0=0.05,v=0.5,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    k0 = ((1-(17/18)**14) * 0.0138 + 0.2*(1-(1-(17/18)**14))) * I_0
    Nation = Qua_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    kB, kA = Nation.kB, Nation.kA

    # t=1 πB for those who sign up， πA for those who don't. We prioritize those who sign up over those who don't
    πB_coef=(1-k0)*(1-m)+x*(1-k0)*m
    πA_coef=(1-x)*(1-k0)*m
    πB = min(m/πB_coef,1)
    πA = (m-πB_coef)/πA_coef if πB==1 else 0

    # two cases at t=2
    if πB<1:
        π2_0_coef= (1-x)*( m*(1-k0)-kA)
        π2_1_coef=(1-πB)*( (1-m)*(1-k0)-kB+ x*(m*(1-k0)-kA) )
        π2_2_coef=   πB *( (1-m)*(1-k0)-kB+ x*(m*(1-k0)-kA) )

        πA0 = min(m/π2_0_coef,1)
        πB0 = min((m-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
        πA1 = 1 if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    else:
        π2_0_coef=(1-πA)*(1-x)*( m*(1-k0)-kA)
        π2_1_coef=   πA *(1-x)*( m*(1-k0)-kA)
        π2_2_coef= (1-m)*(1-k0)-kB+ x*(m*(1-k0)-kA) 
        
        πA0 = min(m/π2_0_coef,1)
        πB0 = 1 if πA0==1 else 0
        πA1 = min((m-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
        πB1 = (m-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

    n_sign = πB*(1+𝜌*(πB1+(1-πB1)*v)) + (1-πB)*(v+πB0*𝜌)
    n_nsign= πA*(1+𝜌*(πA1+(1-πA1)*v)) + (1-πA)*(v+πA0*𝜌)
    return n_sign-n_nsign

def GRBT_Qua(m=0.2,I_0=0.05,𝜌=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,vo=0.5,vn=0.7):
    
    ICphi_sep, ICn_sep = GRBTp1q0_Qua(m=m,I_0=I_0,v=vo/vn,𝜌=𝜌,σo=σo,σn=σn,δo=δo,δn=δn)
    
    #### Fully-Separating Equilibrium ####
    if ICphi_sep>=0 and ICn_sep<=0:
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Qua(1,0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
        D_val = GRBT_evalD_Qua(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
        S3,I3 = GRBT_evalT3_Qua(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=3)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,1,0,ICphi_sep,ICn_sep,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    if ICphi_sep<0:
        #### Partial-Separating Equilibrium. #### 
        # People without mask play mix strategy
        p_res=root(pmix_func_Qua,1,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n),method='broyden1',tol=10e-12)

        if p_res.success and p_res.x>0 and p_res.x<1:
            p_star=p_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Qua(x=p_star,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
            D_val = GRBT_evalD_Qua(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
            S3,I3 = GRBT_evalT3_Qua(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=3)
            ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            
            return m,p_star,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        
        #### Pooling Equlibrium ####
        # No people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Qua(x=0,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
        D_val = GRBT_evalD_Qua(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
        S3,I3 = GRBT_evalT3_Qua(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=3)
        ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud

        
        
    if ICn_sep>0:
        #### Partial-Separating Equilibrium. #### 
        # People having mask play mix strategy
        q_res=root(qmix_func_Qua,0.2,args=(m,I_0,vo/vn,𝜌,σo,σn,𝛿o,𝛿n),method='broyden1',tol=10e-12)

        if q_res.success and q_res.x>0 and q_res.x<1:
            q_star=q_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Qua(x=1,y=q_star,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
            D_val = GRBT_evalD_Qua(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
            S3,I3 = GRBT_evalT3_Qua(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=3)
            ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)

            return m,1,q_star,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        
        #### Pooling Equlibrium ####
        # All people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Qua(x=1,y=1,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn)
        D_val = GRBT_evalD_Qua(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                               S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
        S3,I3 = GRBT_evalT3_Qua(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                            S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,t=3)
        ICphi, ICn = IC_cons(v=vo/vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
    
    
def SRA1(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    k0 = ((1-(17/18)**14) * 0.0138 + 0.2*(1-(1-(17/18)**14))) * I_0
    Nation = Qua_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    kB, kA = Nation.kB, Nation.kA


    π1 = m/(1-k0)

    π2_coef = (1-m)*(1-k0)-kB + m*(1-k0)-kA

    π2 = m/π2_coef

    func = evalD_Qua(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2,
                          σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    S3,I3 = evalT3_Qua(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π2,πB1=π2,πA0=π2,πA1=π2,
                       σo=σo,σn=σn,δo=δo,δn=δn,t=3)
    Uphi,Un,Ud = Udiff(vo=0.5,vn=0.7,𝜌=1,πB=π1,πA=π1,πA0=π2,πB0=π2,πA1=π2,πB1=π2)

    return {'func':func,'π1':π1,'π2':π2,'S3':S3,'I3':I3,'Uphi':Uphi,'Un':Un,'Ud':Ud}

def SRA2(m=0.1,I_0=0.01,σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    k0 = ((1-(17/18)**14) * 0.0138 + 0.2*(1-(1-(17/18)**14))) * I_0
    Nation = Qua_SIRD(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    kB, kA = Nation.kB, Nation.kA

    π1 = m/(1-k0)

    π20_coef = (1-π1)*( (1-m)*(1-k0)-kB + m*(1-k0)-kA )
    π21_coef =    π1 *( (1-m)*(1-k0)-kB + m*(1-k0)-kA )

    π20 = min(m/π20_coef,1) #
    π21 = (m-π20_coef)/π21_coef if π20==1 else 0

    func = evalD_Qua(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21,
                          σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    S3,I3 = evalT3_Qua(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πB0=π20,πB1=π21,πA0=π20,πA1=π21,
                       σo=σo,σn=σn,δo=δo,δn=δn,t=3)
    Uphi,Un,Ud = Udiff(vo=0.5,vn=0.7,𝜌=1,πB=π1,πA=π1,πA0=π20,πB0=π20,πA1=π21,πB1=π21)

    return {'func':func,'π1':π1,'π20':π20,'π21':π21,'S3':S3,'I3':I3,'Uphi':Uphi,'Un':Un,'Ud':Ud}