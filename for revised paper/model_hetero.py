from scipy.optimize import root
import numpy as np
import pandas as pd

'''
T3_Hetero
evaldeath_Hetero
evaldeath_Hetero_T3
IC_cons_Hetero


pmix_Hetero
qmix_Hetero
GRBTp1q0_Hetero
Mix_computeProb_Hetero
GRBT_evalD_Hetero
GRBT_evalT3_Hetero
GRBT_Hetero


--------------------------------------------------
Other Functions for Comparison

'''




class T3_Hetero:
    
    def __init__(self,S=0.9,        # initial susceptible
                      I=0.1,        # initial infected
                      R=0,          # initial recovered
                      D=0,          # initial died
                      𝛽L=2.4/(18/14),# basic transmission rate. R0=2.4 and it takes 18 days to leave I state in average.
                                    # Furthermore, a time unit is 14 days here.
                      𝛽H=4.8/(18/14),
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

                                         
        self.S    = np.array([S,0,0,0,0,0])
        self.I    = np.array([I,0,0,0,0,0])
        self.R, self.D  = R, D
        self.𝛽L, self.𝛽H, self.𝛾, self.𝛼 = 𝛽L, 𝛽H, 𝛾, 𝛼
        self.σo, self.σn, self.δo, self.δn = σo, σn, δo, δn

        self.T, self.𝜋0 = T, 𝜋0
        self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1 = πB, πA, πB0, πA0, πB1, πA1
        
    def evaluate_change(self):
        T = self.T
        𝛽L, 𝛽H, 𝛾, 𝛼 = self.𝛽L, self.𝛽H, self.𝛾, self.𝛼
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        𝜋0, πB, πA, πB0, πA0, πB1, πA1 = self.𝜋0, self.πB, self.πA, self.πB0, self.πA0, self.πB1, self.πA1

        if T==0:
            # population distribution after issuing mask
            transition  = np.array([[1-𝜋0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [  𝜋0,0,0,0,0,0]])

        
        if T==1:
            ##### Compute number of death in period 1 for computing probability of receiving a new mask
            self.dB, self.dA = 𝛾 * 𝛼 * self.I[0], 𝛾 * 𝛼 * self.I[4]
            # population distribution after issuing mask
            transition =  np.array([[1-πB,0,0,0,   0,0],
                                    [   0,0,0,0,   0,0],
                                    [  πB,0,0,0,   0,0],
                                    [   0,0,0,0,   0,0],
                                    [   0,0,0,0,1-πA,0],
                                    [   0,0,0,0,  πA,0]])

                                      


        elif T==2:
            # population distribution after issuing mask
            transition = np.array([[1-πB0,    0,0,    0,    0,      0],
                                   [    0,1-πB1,0,    0,    0,      0],
                                   [  πB0,  πB1,0,    0,    0,      0],
                                   [    0,    0,0,1-πA0,    0,      0],
                                   [    0,    0,0,    0,1-πA1,      0],
                                   [    0,    0,0,  πA0,  πA1,      0]])

            
        elif T>=3:
            # Begining from period 3, every one will receive a new mask
            transition = np.array([[0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [1,1,1,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,1,1,1]])

        S_mask = transition.dot(self.S) # 6x1
        I_mask = transition.dot(self.I) # 6x1
        
        # masking state: ϕ_L o_L n_L | ϕ_H o_H n_H
        inward_L = 𝛽L*np.array([1,(1-σo),(1-σn)])
        inward_H = 𝛽H*np.array([1,(1-σo),(1-σn)])            
        𝛽0 = np.outer([1,(1-δo),(1-δn),1,(1-δo),(1-δn)], np.append(inward_L,inward_H))
        # transmission
        dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
        
        # people leave from state I
        dR = 𝛾 * (1-𝛼) * I_mask
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = self.R + sum(dR)
        nD = self.D + sum(dD)

        # transition of masking state
        # new masks deteriate while old mask are dicarded 
        transition_mask = np.array([[1,1,0,0,0,0],
                                    [0,0,1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,1,1,0],
                                    [0,0,0,0,0,1],
                                    [0,0,0,0,0,0]])
        
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



def evaldeath_Hetero(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=10,π0=0.2,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        # population distribution after issuing mask
        if T==0:
            # population distribution after issuing mask
            transition  = np.array([[1-π0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [   0,0,0,0,0,0],
                                    [  π0,0,0,0,0,0]])

        
        if T==1:
            # population distribution after issuing mask
            transition =  np.array([[1-πB,0,0,0,   0,0],
                                    [   0,0,0,0,   0,0],
                                    [  πB,0,0,0,   0,0],
                                    [   0,0,0,0,   0,0],
                                    [   0,0,0,0,1-πA,0],
                                    [   0,0,0,0,  πA,0]])

                                      


        elif T==2:
            # population distribution after issuing mask
            transition = np.array([[1-πB0,    0,0,    0,    0,      0],
                                   [    0,1-πB1,0,    0,    0,      0],
                                   [  πB0,  πB1,0,    0,    0,      0],
                                   [    0,    0,0,1-πA0,    0,      0],
                                   [    0,    0,0,    0,1-πA1,      0],
                                   [    0,    0,0,  πA0,  πA1,      0]])

            
        elif T>=3:
            # Begining from period 3, every one will receive a new mask
            transition = np.array([[0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [1,1,1,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,1,1,1]])

        S_mask = transition.dot(S) # 6x1
        I_mask = transition.dot(I) # 6x1
        
        # masking state: ϕ_L o_L n_L | ϕ_H o_H n_H
        inward_L = 𝛽L*np.array([1,(1-σo),(1-σn)])
        inward_H = 𝛽H*np.array([1,(1-σo),(1-σn)])            
        𝛽0 = np.outer([1,(1-δo),(1-δn),1,(1-δo),(1-δn)], np.append(inward_L,inward_H))
        # transmission
        dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
        
        # people leaving from state I
        dR = 𝛾 * (1-𝛼) * I_mask
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = R + sum(dR)
        nD = D + sum(dD)

        # transition of masking state
        # new masks deteriate while old mask are dicarded 
        transition_mask = np.array([[1,1,0,0,0,0],
                                    [0,0,1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,1,1,0],
                                    [0,0,0,0,0,1],
                                    [0,0,0,0,0,0]])
        
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        
        S,I,R,D=nS,nI,nR,nD
        
        T=T+1
    
    return(D)

def evaldeath_Hetero_T3(πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=3,π0=0.2,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        # population distribution after issuing mask
        if T==0:
            # population distribution after issuing mask
            transition  = np.array([[1-π0,0,0,0,0,0],
                                    [    0,0,0,0,0,0],
                                    [    0,0,0,0,0,0],
                                    [    0,0,0,0,0,0],
                                    [    0,0,0,0,0,0],
                                    [  π0,0,0,0,0,0]])

        
        if T==1:
            # population distribution after issuing mask
            transition =  np.array([[1-πB,0,0,0,   0,0],
                                    [   0,0,0,0,   0,0],
                                    [  πB,0,0,0,   0,0],
                                    [   0,0,0,0,   0,0],
                                    [   0,0,0,0,1-πA,0],
                                    [   0,0,0,0,  πA,0]])

                                      


        elif T==2:
            # population distribution after issuing mask
            transition = np.array([[1-πB0,    0,0,    0,    0,      0],
                                   [    0,1-πB1,0,    0,    0,      0],
                                   [  πB0,  πB1,0,    0,    0,      0],
                                   [    0,    0,0,1-πA0,    0,      0],
                                   [    0,    0,0,    0,1-πA1,      0],
                                   [    0,    0,0,  πA0,  πA1,      0]])
            
        elif T>=3:
            # Begining from period 3, every one will receive a new mask
            transition = np.array([[0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [1,1,1,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,1,1,1]])


        S_mask = transition.dot(S) # 6x1
        I_mask = transition.dot(I) # 6x1
        
        # masking state: ϕ_L o_L n_L | ϕ_H o_H n_H
        inward_L = 𝛽L*np.array([1,(1-σo),(1-σn)])
        inward_H = 𝛽H*np.array([1,(1-σo),(1-σn)])            
        𝛽0 = np.outer([1,(1-δo),(1-δn),1,(1-δo),(1-δn)], np.append(inward_L,inward_H))
        # transmission
        dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
        
        # people leaving from state I
        dR = 𝛾 * (1-𝛼) * I_mask
        dD = 𝛾 * 𝛼 * I_mask 

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = R + sum(dR)
        nD = D + sum(dD)

        # transition of masking state
        # new masks deteriate while old mask are dicarded 
        transition_mask = np.array([[1,1,0,0,0,0],
                                    [0,0,1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,1,1,0],
                                    [0,0,0,0,0,1],
                                    [0,0,0,0,0,0]])
        
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        
        S,I,R,D=nS,nI,nR,nD
        
        T=T+1
    
    return sum(S),sum(I)


def IC_cons_Hetero(v=0.5,𝜌L=1,𝜌H=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    phi_sign = 𝜋B*(1+𝜌L*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌L
    phi_nsign= 𝜋A*(1+𝜌L*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*𝜋A0*𝜌L
    n_sign   = 𝜋B*(1+𝜌H*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*(v+𝜋B0*𝜌H)
    n_nsign  = 𝜋A*(1+𝜌H*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌H)

    ICphi = phi_sign-phi_nsign
    ICn   = n_sign-n_nsign
    return ICphi, ICn


def Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=1,𝜌H=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    
    v=vo/vn
    phi_sign = 𝜋B*(1+𝜌L*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌L
    phi_nsign= 𝜋A*(1+𝜌L*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*𝜋A0*𝜌L
    n_sign   = 𝜋B*(1+𝜌H*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*(v+𝜋B0*𝜌H)
    n_nsign  = 𝜋A*(1+𝜌H*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌H)
    
    Uphi =    𝜌L * vn * max(phi_sign,phi_nsign)
    Un = vn + 𝜌H * vn * max(n_sign,n_nsign)
    
    return Uphi, Un, Un-Uphi




def pmix_Hetero(x,m,I_0=0.05,v=0.5,𝜌L=1,σo=0.5,σn=0.7,𝛿o=0.5,𝛿n=0.7,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14)):

     # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,𝛿o=𝛿o,𝛿n=𝛿n,𝛽L=𝛽L,𝛽H=𝛽H)
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

    ICphi, ICn = IC_cons_Hetero(v=v,𝜌L=𝜌L,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)

    return ICphi


def qmix_Hetero(x,m,I_0=0.05,v=0.5,𝜌H=1,σo=0.5,σn=0.7,𝛿o=0.5,𝛿n=0.7,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14)):

     # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,𝛿o=𝛿o,𝛿n=𝛿n,𝛽L=𝛽L,𝛽H=𝛽H)
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

    ICphi, ICn = IC_cons_Hetero(v=v,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)

    return ICn

def GRBTp1q0_Hetero(m=0.2,I_0=0.05,v=0.5/0.7,𝜌L=1,𝜌H=1,σo=0.5,σn=0.7,𝛿o=0.5,𝛿n=0.7,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14)):

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,𝛿o=𝛿o,𝛿n=𝛿n,𝛽L=𝛽L,𝛽H=𝛽H)
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

    ICphi, ICn = IC_cons_Hetero(v=v,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
    
    return ICphi,ICn


def Mix_computeProb_Hetero(x,y,m,I_0=0.05,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14)):

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,σo=σo,σn=σn,𝛿o=𝛿o,𝛿n=𝛿n,𝛽L=𝛽L,𝛽H=𝛽H)
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



def GRBT_evalD_Hetero(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=10,
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
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            # masking state: ϕ_L n_L ϕ_L n_L | o_H n_H o_H n_H
            inward_L = 𝛽L*np.array([1     ,(1-σn),1     ,(1-σn)])
            inward_H = 𝛽H*np.array([(1-σo),(1-σn),(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # transition of masking state
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],  
                                        [0,0,0,0,0,1,0,0],   
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0], 
                                        [0,0,0,0,0,0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            # 先登記是否要買口罩
            signup= np.array([[1-p,0,0,0,  0,0,0,0],
                              [  0,0,0,0,  0,0,0,0],
                              [  p,0,0,0,  0,0,0,0],
                              [  0,0,0,0,  0,0,0,0],
                              [  0,0,0,0,1-q,0,0,0],
                              [  0,0,0,0,  0,0,0,0],
                              [  0,0,0,0,  q,0,0,0],
                              [  0,0,0,0,  0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)

            # 再來根據登記與否發口罩
            transition_1 =  np.array([[1-πA,0,   0,0,   0,0,   0,0],
                                      [  πA,0,   0,0,   0,0,   0,0],
                                      [   0,0,1-πB,0,   0,0,   0,0],
                                      [   0,0,  πB,0,   0,0,   0,0],
                                      [   0,0,   0,0,1-πA,0,   0,0],
                                      [   0,0,   0,0,  πA,0,   0,0],
                                      [   0,0,   0,0,   0,0,1-πB,0],
                                      [   0,0,   0,0,   0,0,  πB,0]])

                                    
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            # masking state: ϕ_L n_L ϕ_L n_L | o_H n_H o_H n_H
            inward_L = 𝛽L*np.array([1     ,(1-σn),1     ,(1-σn)])
            inward_H = 𝛽H*np.array([(1-σo),(1-σn),(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            # population distribution after issuing mask
            
            transition_2 = np.array([[1-πA0,    0,1-πB0,    0,    0,    0,    0,    0],
                                     [    0,1-πA1,    0,1-πB1,    0,    0,    0,    0],
                                     [  πA0,  πA1,  πB0,  πB1,    0,    0,    0,    0],
                                     [    0,    0,    0,    0,1-πA0,    0,1-πB0,    0],
                                     [    0,    0,    0,    0,    0,1-πA1,    0,1-πB1],
                                     [    0,    0,    0,    0,  πA0,  πA1,  πB0,  πB1]])
            S_mask = transition_2.dot(S) # 6x1
            I_mask = transition_2.dot(I) # 6x1

            # masking state: ϕ_L o_L n_L | ϕ_H o_H n_H
            inward_L = 𝛽L*np.array([1,(1-σo),(1-σn)])
            inward_H = 𝛽H*np.array([1,(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δo),(1-δn),1,(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

            # people leave from state I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            # transition of masking state
            # new masks deteriate while old mask are dicarded 
            transition_mask = np.array([[1,1,0,0,0,0],
                                        [0,0,1,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,0,1,1,0],
                                        [0,0,0,0,0,1],
                                        [0,0,0,0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T>=3:
            # Begining from period 3, every one will receive a new mask
            transition = np.array([[0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [1,1,1,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,1,1,1]])

            S_mask = transition.dot(S) # 6x1
            I_mask = transition.dot(I) # 6x1

            # masking state: ϕ_L o_L n_L | ϕ_H o_H n_H
            inward_L = 𝛽L*np.array([1,(1-σo),(1-σn)])
            inward_H = 𝛽H*np.array([1,(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δo),(1-δn),1,(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

            # people leave from state I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # transition of masking state
            # new masks deteriate while old mask are dicarded 
            transition_mask = np.array([[1,1,0,0,0,0],
                                        [0,0,1,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,0,1,1,0],
                                        [0,0,0,0,0,1],
                                        [0,0,0,0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD


        T=T+1
    
    return(D)

def GRBT_evalT3_Hetero(p=1,q=0,
                   πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2,
                   π0=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),𝛾=1-(17/18)**14,𝛼=0.0138,
                   T=0,t=10,
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
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [  π0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0],
                                      [   0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            # masking state: ϕ_L n_L ϕ_L n_L | o_H n_H o_H n_H
            inward_L = 𝛽L*np.array([1     ,(1-σn),1     ,(1-σn)])
            inward_H = 𝛽H*np.array([(1-σo),(1-σn),(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # transition of masking state
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],  
                                        [0,0,0,0,0,1,0,0],   
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0], 
                                        [0,0,0,0,0,0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            # 先登記是否要買口罩
            signup= np.array([[1-p,0,0,0,  0,0,0,0],
                              [  0,0,0,0,  0,0,0,0],
                              [  p,0,0,0,  0,0,0,0],
                              [  0,0,0,0,  0,0,0,0],
                              [  0,0,0,0,1-q,0,0,0],
                              [  0,0,0,0,  0,0,0,0],
                              [  0,0,0,0,  q,0,0,0],
                              [  0,0,0,0,  0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)

            # 再來根據登記與否發口罩
            transition_1 =  np.array([[1-πA,0,   0,0,   0,0,   0,0],
                                      [  πA,0,   0,0,   0,0,   0,0],
                                      [   0,0,1-πB,0,   0,0,   0,0],
                                      [   0,0,  πB,0,   0,0,   0,0],
                                      [   0,0,   0,0,1-πA,0,   0,0],
                                      [   0,0,   0,0,  πA,0,   0,0],
                                      [   0,0,   0,0,   0,0,1-πB,0],
                                      [   0,0,   0,0,   0,0,  πB,0]])

                                    
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            # masking state: ϕ_L n_L ϕ_L n_L | o_H n_H o_H n_H
            inward_L = 𝛽L*np.array([1     ,(1-σn),1     ,(1-σn)])
            inward_H = 𝛽H*np.array([(1-σo),(1-σn),(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
            dR = 𝛾 * (1-𝛼) * I_mask 
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            # population distribution after issuing mask
            
            transition_2 = np.array([[1-πA0,    0,1-πB0,    0,    0,    0,    0,    0],
                                     [    0,1-πA1,    0,1-πB1,    0,    0,    0,    0],
                                     [  πA0,  πA1,  πB0,  πB1,    0,    0,    0,    0],
                                     [    0,    0,    0,    0,1-πA0,    0,1-πB0,    0],
                                     [    0,    0,    0,    0,    0,1-πA1,    0,1-πB1],
                                     [    0,    0,    0,    0,  πA0,  πA1,  πB0,  πB1]])
            S_mask = transition_2.dot(S) # 6x1
            I_mask = transition_2.dot(I) # 6x1

            # masking state: ϕ_L o_L n_L | ϕ_H o_H n_H
            inward_L = 𝛽L*np.array([1,(1-σo),(1-σn)])
            inward_H = 𝛽H*np.array([1,(1-σo),(1-σn)])            
            𝛽0 = np.outer([1,(1-δo),(1-δn),1,(1-δo),(1-δn)], np.append(inward_L,inward_H))
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

            # people leave from state I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask 

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            # transition of masking state
            # new masks deteriate while old mask are dicarded 
            transition_mask = np.array([[1,1,0,0,0,0],
                                        [0,0,1,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,0,1,1,0],
                                        [0,0,0,0,0,1],
                                        [0,0,0,0,0,0]])

            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD


        T=T+1
    
    return sum(S),sum(I)


def GRBT_Hetero(m=0.2,I_0=0.01,vo=0.5,vn=0.7,𝜌L=1,𝜌H=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛽L=14*2.4/18,𝛽H=14*4.8/18):
    
    ICphi_sep, ICn_sep = GRBTp1q0_Hetero(m=m,I_0=I_0,v=vo/vn,𝜌L=𝜌L,𝜌H=𝜌H,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
    
    #### Fully-Separating Equilibrium ####
    if ICphi_sep>=0 and ICn_sep<=0:
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(1,0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
        D_val = GRBT_evalD_Hetero(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                  S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
        S3,I3 = GRBT_evalT3_Hetero(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=3)
        Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,1,0,ICphi_sep,ICn_sep,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    if ICphi_sep<0:
        #### Partial-Separating Equilibrium. #### 
        # People without mask play mix strategy
        p_res=root(pmix_Hetero,0.9,args=(m,I_0,vo/vn,𝜌L,σo,σn,𝛿o,𝛿n,𝛽L,𝛽H),method='krylov',tol=10e-12)

        if p_res.success and p_res.x>0 and p_res.x<1:
            p_star=p_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=p_star,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
            D_val = GRBT_evalD_Hetero(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                      S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
            S3,I3 = GRBT_evalT3_Hetero(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                       S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=3)
            ICphi, ICn = IC_cons_Hetero(v=vo/vn,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            
            return m,p_star,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
        
        #### Pooling Equlibrium ####
        # No people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=0,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
        D_val = GRBT_evalD_Hetero(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                  S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
        S3,I3 = GRBT_evalT3_Hetero(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=3)
        ICphi, ICn = IC_cons_Hetero(v=vo/vn,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud

        
        
    if ICn_sep>0:
        #### Partial-Separating Equilibrium. #### 
        # People having mask play mix strategy
        q_res=root(qmix_Hetero,0.2,args=(m,I_0,vo/vn,𝜌H,σo,σn,𝛿o,𝛿n,𝛽L,𝛽H),method='excitingmixing',tol=10e-12)

        if q_res.success and q_res.x>0 and q_res.x<1:
            q_star=q_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=1,y=q_star,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
            D_val = GRBT_evalD_Hetero(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                      S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
            S3,I3 = GRBT_evalT3_Hetero(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                       S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=3)
            ICphi, ICn = IC_cons_Hetero(v=vo/vn,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
            Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)

            return m,1,q_star,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud 
        
        #### Pooling Equlibrium ####
        # All people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=1,y=1,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
        D_val = GRBT_evalD_Hetero(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                  S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
        S3,I3 = GRBT_evalT3_Hetero(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                   S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=3)
        ICphi, ICn = IC_cons_Hetero(v=vo/vn,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=𝜌L,𝜌H=𝜌H,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1)
        
        return m,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,S3,I3,D_val,Uphi,Un,Ud
    

    
def SRA1_Hetero(m=0.1,I_0=0.01,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    
    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    π1 = m/(1-d0)

    π2_coef = (1-m)*(1-d0)-dB + m*(1-d0)-dA

    π2 =m/π2_coef
    
    func = evaldeath_Hetero(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πA0=π2,πB0=π2,πA1=π2,πB1=π2,
                            𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=1,𝜌H=1,πB=π1,πA=π1,πA0=π2,πB0=π2,πA1=π2,πB1=π2)

    return {'func':func,'π1':π1,'π2':π2,'Uphi':Uphi,'Un':Un,'Ud':Ud}   
    
    
    
def SRA2_Hetero(m=0.1,I_0=0.01,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    π1 = m/(1-d0)

    π20_coef = (1-π1)*( (1-m)*(1-d0)-dB + m*(1-d0)-dA )
    π21_coef =    π1 *( (1-m)*(1-d0)-dB + m*(1-d0)-dA )

    π20 = np.minimum(m/π20_coef,1) 
    π21 = (m-π20_coef)/π21_coef if π20==1 else 0
    
    func = evaldeath_Hetero(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πA0=π20,πB0=π20,πA1=π21,πB1=π21,
                            𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn,t=300)
    Uphi,Un,Ud =Udiff_Hetero(vo=0.5,vn=0.7,𝜌L=1,𝜌H=1,πB=π1,πA=π1,πA0=π20,πB0=π20,πA1=π21,πB1=π21)

    return {'func':func,'π1':π1,'π20':π20,'π21':π21,'Uphi':Uphi,'Un':Un,'Ud':Ud}    
    
    
    
    
    
    
    
    
    
#===========================================================================================================#
#===========================================================================================================#
#===========================================================================================================#


#===========================================================================================================#
#===========================================================================================================#
#===========================================================================================================#
    
    
    
    
def GRBT_forcomparison_Hetero(m=0.2,I_0=0.01,v=0.5/0.7,𝜌L=1,𝜌H=1,σo=0.5,σn=0.7,δo=0.5,δn=0.7,𝛽L=14*2.4/18,𝛽H=14*4.8/18):
    
    ICphi_sep, ICn_sep = GRBTp1q0_Hetero(m=m,I_0=I_0,v=v,𝜌L=𝜌L,𝜌H=𝜌H,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
    
    #### Fully-Separating Equilibrium ####
    if ICphi_sep>=0 and ICn_sep<=0:
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(1,0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
        D_val = GRBT_evalD_Hetero(p=1,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                  S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
        
        return D_val
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    if ICphi_sep<0:
        #### Partial-Separating Equilibrium. #### 
        # People without mask play mix strategy
        p_res=root(pmix_Hetero,0.9,args=(m,I_0,v,𝜌L,σo,σn,𝛿o,𝛿n,𝛽L,𝛽H),method='krylov',tol=10e-12)

        if p_res.success and p_res.x>0 and p_res.x<1:
            p_star=p_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=p_star,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
            D_val = GRBT_evalD_Hetero(p=p_star,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                      S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
            
            return D_val
        
        #### Pooling Equlibrium ####
        # No people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=0,y=0,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
        D_val = GRBT_evalD_Hetero(p=0,q=0,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                  S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
        
        return D_val

        
        
    if ICn_sep>0:
        #### Partial-Separating Equilibrium. #### 
        # People having mask play mix strategy
        q_res=root(qmix_Hetero,0.2,args=(m,I_0,v,𝜌H,σo,σn,𝛿o,𝛿n,𝛽L,𝛽H),method='excitingmixing',tol=10e-12)

        if q_res.success and q_res.x>0 and q_res.x<1:
            q_star=q_res.x
            πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=1,y=q_star,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
            D_val = GRBT_evalD_Hetero(p=1,q=q_star,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                      S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)

            return D_val 
        
        #### Pooling Equlibrium ####
        # All people sign equilibrium
        πB,πA,πA0,πB0,πA1,πB1 = Mix_computeProb_Hetero(x=1,y=1,m=m,I_0=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H)
        D_val = GRBT_evalD_Hetero(p=1,q=1,π0=m,πB=πB,πA=πA,πA0=πA0,πB0=πB0,πA1=πA1,πB1=πB1,
                                  S=1-I_0,I=I_0,σo=σo,σn=σn,δo=δo,δn=δn,𝛽L=𝛽L,𝛽H=𝛽H,t=300)
        
        return D_val
    
    
    
    
def SRA2_forcomparison_Hetero(m=0.1,I_0=0.01,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    π1 = m/(1-d0)

    π20_coef = (1-π1)*( (1-m)*(1-d0)-dB + m*(1-d0)-dA )
    π21_coef =    π1 *( (1-m)*(1-d0)-dB + m*(1-d0)-dA )

    π20 = np.minimum(m/π20_coef,1) 
    π21 = (m-π20_coef)/π21_coef if π20==1 else 0
    
    func = evaldeath_Hetero(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πA0=π20,πB0=π20,πA1=π21,πB1=π21,
                            𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn,t=300)

    return func    

def SRA1_forcomparison_Hetero(m=0.1,I_0=0.01,𝛽L=1.2/(18/14),𝛽H=2.4/(18/14),σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    
    # numbers of death at the initial two periods 
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = T3_Hetero(S=1-I_0,I=I_0,π0=m,πB=0,πA=0,𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn)
    Nation.severalupdates(2)
    dB, dA = Nation.dB, Nation.dA


    π1 = m/(1-d0)

    π2_coef = (1-m)*(1-d0)-dB + m*(1-d0)-dA

    π2 =m/π2_coef
    
    func = evaldeath_Hetero(S=1-I_0,I=I_0,π0=m,πB=π1,πA=π1,πA0=π2,πB0=π2,πA1=π2,πB1=π2,
                            𝛽L=𝛽L,𝛽H=𝛽H,σo=σo,σn=σn,δo=δo,δn=δn,t=300)

    return func   


def SRA1vsSRA2_Hetero(m,k):
    
    𝛽H_set= 2.4/(18/14)
    𝛽L_set= k * 𝛽H_set
    
    D_SRA1= SRA1_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set)
    D_SRA2= SRA2_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set)
    return D_SRA2-D_SRA1
    
def SRA2vsGRBT_Hetero(m,k):
    
    𝛽H_set= 2.4/(18/14)
    𝛽L_set= k * 𝛽H_set
    
    D_SRA2= SRA2_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set)
    D_GRBT= GRBT_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set)
    
    return D_GRBT-D_SRA2

def SRA1vs2vsGRBT_Hetero(m,k,σo=0.5,σn=0.7,δo=0.5,δn=0.7):
    
    𝛽H_set= 2.4/(18/14)
    𝛽L_set= k * 𝛽H_set
    
    D_SRA1= SRA1_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set,σo=σo,σn=σn,δo=δo,δn=δn)
    D_SRA2= SRA2_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set,σo=σo,σn=σn,δo=δo,δn=δn)
    D_GRBT= GRBT_forcomparison_Hetero(m,𝛽L=𝛽L_set,𝛽H=𝛽H_set,σo=σo,σn=σn,δo=δo,δn=δn)
    
    return D_SRA1,D_SRA2,D_GRBT