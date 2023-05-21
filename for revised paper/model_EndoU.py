import numpy as np
import pandas as pd
from math import log
from scipy.optimize import bisect


class Params():
    def __init__(self,
                 S=0.99,I=0.01,R=0,D=0,       # initial SIRD
                 𝛽=2.4/(18/14),               # dynamic parameters: transmission rate
                 𝛾=1-(17/18)**14,             # dynamic parameters: rate of moving out I
                 𝛼=0.0138,                    # dynamic parameters: death rate
                 
                 σn=0.7,δn=0.7,               # new mask inward/outward protection rate
                 σo=0.5,δo=0.5,               # old mask inward/outward protection rate
                 p=1,q=0,                     # probs of early participation
                 m=0.2,                       # mask supply level
                 protect_set="benchmark",     # mask protection rate setting
                 mask_supply="benchmark"):    # mask supply pattern
        
        self.S, self.I, self.R, self.D = S, I, R, D
        self.𝛽, self.𝛾, self.𝛼 = 𝛽, 𝛾, 𝛼
        self.p, self.q = p, q
        self.m = m
        self.protect_set = protect_set
        self.mask_supply = mask_supply
    
    def supply_pattern(self):
        m=self.m
        mask_supply = self.mask_supply
        
        if mask_supply=="benchmark":
            self.m0, self.m1, self.m2= m,m,m
        elif mask_supply=="growth":
            self.m0, self.m1, self.m2= m,m,1.15*m
        elif mask_supply=="init_stock":
            self.m0, self.m1, self.m2= 1.15*m,m,m
    
    def mask_protection(self):
        protect_set = self.protect_set
        
        if protect_set=="benchmark":
            self.σn, self.δn = 0.7, 0.7
            self.σo, self.δo = 0.5, 0.5
        elif protect_set=="lower_inward":
            self.σn, self.δn = 0.6, 0.7
            self.σo, self.δo = 0.4, 0.5
        elif protect_set=="similar_prot":
            self.σn, self.δn = 0.7, 0.7
            self.σo, self.δo = 0.69, 0.69
        
class GetMaskProbs(Params):
    '''
        Given the set of parameters, calculate the probs of getting new mask under certain mechanism
    '''
    
    def __init__(self,mech):
        super().__init__()
        self.mech=mech
        
    def calculate_probs(self):
        
        # calculate the mask supply and protection rate
        self.supply_pattern()
        self.mask_protection()
        
        # parameters
        S, I = self.S, self.I
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        m0, m1, m2 = self.m0, self.m1, self.m2
        p, q, = self.p, self.q
        σn,δn = self.σn, self.δn
        
        # numbers of death at the second periods 
        d0 = 𝛾 * 𝛼 * I
        dA = 𝛾 * 𝛼 * ( (1-𝛾)*   m0 *I + 𝛽*   m0 *(1-σn)*S*(m0*(1-δn)*I+(1-m0)*I) )
        dB = 𝛾 * 𝛼 * ( (1-𝛾)*(1-m0)*I + 𝛽*(1-m0)*       S*(m0*(1-δn)*I+(1-m0)*I) )
        
        ############ GRBT machanism ############
        if self.mech=="GRBT":
            
            # t=1
            πB_coef=   p *(1-m0)*(1-d0) +    q *m0*(1-d0)
            πA_coef=(1-p)*(1-m0)*(1-d0) + (1-q)*m0*(1-d0)
            πB = min(m1/πB_coef,1) if πB_coef>0 else 1
            πA = (m1-πB_coef)/πA_coef if πB==1 else 0

            # t=2 
            if πB<1:
                π2_0_coef=        (1-p)*((1-m0)*(1-d0)-dB) + (1-q)*(m0*(1-d0)-dA)
                π2_1_coef=(1-πB)*(   p *((1-m0)*(1-d0)-dB) +    q *(m0*(1-d0)-dA)) 
                π2_2_coef=   πB *(   p *((1-m0)*(1-d0)-dB) +    q *(m0*(1-d0)-dA))

                πA0 = min(m2/π2_0_coef,1) if π2_0_coef>0 else 1
                πB0 = min((m2-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
                πA1 = 1 if πB0==1 else 0
                πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

            else:
                π2_0_coef=(1-πA)*((1-p)*((1-m0)*(1-d0)-dB) + (1-q)*(m0*(1-d0)-dA))
                π2_1_coef=   πA *((1-p)*((1-m0)*(1-d0)-dB) + (1-q)*(m0*(1-d0)-dA))
                π2_2_coef=           p *((1-m0)*(1-d0)-dB) +    q *(m0*(1-d0)-dA)


                πA0 = min(m2/π2_0_coef,1) if π2_0_coef>0 else 1
                πB0 = 1 if πA0==1 else 0
                πA1 = min((m2-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
                πB1 = (m2-π2_0_coef-π2_1_coef)/π2_2_coef if πA1==1 else 0

            self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1= πB, πA, πA0, πB0, πA1, πB1
        
        ############ SRA-I machanism ############
        elif self.mech=="SRA1":
            
            # t=1
            π1 = m1/(1-d0)
            # t=2 
            π2_coef = (1-m0)*(1-d0)-dB + m0*(1-d0)-dA
            π2 = m2/π2_coef
            
            πB=πA=π1
            πA0=πB0=πA1=πB1 = π2

            self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1= πB, πA, πA0, πB0, πA1, πB1
        
        ############ SRA-II machanism ############
        elif self.mech=="SRA2":
            
            # t=1
            π1 = m1/(1-d0)
            # t=2 
            π20_coef = (1-π1)*((1-m0)*(1-d0)-dB + m0*(1-d0)-dA)
            π21_coef =    π1 *((1-m0)*(1-d0)-dB + m0*(1-d0)-dA)
            
            π20 = min(m2/π20_coef,1)
            π21 = (m2-π20_coef)/π21_coef if π20==1 else 0
            
            πB=πA=π1
            πA0=πB0=π20
            πA1=πB1=π21

            self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1= πB, πA, πA0, πB0, πA1, πB1
    
        

class Dynamics(GetMaskProbs):
    
    def __init__(self,m,mech,
                 mask_supply="benchmark",
                 protect_set="benchmark"):
        
        super().__init__(mech)
        self.m=m
        self.mask_supply=mask_supply
        self.protect_set=protect_set
        
    
    def evalD(self):
        
        # calculate the prob of getting a mask
        self.calculate_probs()
        
        # initialize the parameters
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        σn, δn = self.σn, self.δn
        σo, δo = self.σo, self.δo
        πB, πA, πA0, πB0, πA1, πB1 = self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1
        p, q = self.p, self.q
        
        S= np.array([(1-self.m0)*self.S,0,0,0,self.m0*self.S,0,0,0])
        I= np.array([(1-self.m0)*self.I,0,0,0,self.m0*self.I,0,0,0])
        R, D = self.R, self.D
        
        t=0
        
        for _ in range(300):
            
            if t==0:

                S_mask = S # 8x1
                I_mask = I # 8x1

                # masking state depends on wether to reserve mask or not
                contagious_vector = [1,(1-δn),1,(1-δn),(1-δn),(1-δn),(1-δo),(1-δn)]
                protective_vector = [1,(1-σn),1,(1-σn),(1-σn),(1-σn),(1-σo),(1-σn)]
                matrix = np.outer(contagious_vector,protective_vector)
                𝛽0 = 𝛽 * matrix
                # transmission
                dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
                # calculate prob of becoming infectious
                P1ϕ = 𝛽*I_mask.dot(contagious_vector)
                P1o = (1-σo)*P1ϕ
                P1n = (1-σn)*P1ϕ
                
        
            if t==1:
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

                # masking state depends on wether to reserve mask or not
                contagious_vector = [1,(1-δn),1,(1-δn),(1-δo),(1-δn),(1-δo),(1-δn)]
                protective_vector = [1,(1-σn),1,(1-σn),(1-σo),(1-σn),(1-σo),(1-σn)]
                matrix = np.outer(contagious_vector,protective_vector)
                𝛽0 = 𝛽 * matrix
                # transmission
                dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
                # calculate prob of becoming infectious
                P2ϕ = 𝛽*I_mask.dot(contagious_vector)
                P2o = (1-σo)*P2ϕ
                P2n = (1-σn)*P2ϕ

            elif t==2:
                # masking state: ϕ n
                transition_2 = np.array([[1-𝜋A0,    0,1-𝜋B0,    0,1-𝜋A0,    0,1-𝜋B0,    0],
                                         [    0,1-𝜋A1,    0,1-𝜋B1,    0,1-𝜋A1,    0,1-𝜋B1],
                                         [  𝜋A0,  𝜋A1,  𝜋B0,  𝜋B1,  𝜋A0,  𝜋A1,  𝜋B0,  𝜋B1]])
                S_mask = transition_2.dot(S) # 3x1
                I_mask = transition_2.dot(I) # 3x1

                # masking state: ϕ n
                contagious_vector = [1,(1-δo),(1-δn)]
                protective_vector = [1,(1-σo),(1-σn)]
                matrix = np.outer(contagious_vector,protective_vector)
                𝛽0 = 𝛽 * matrix
                # 因為 interaction 改變狀態
                dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
                self.t2SI=sum(dS)
                # calculate prob of becoming infectious
                P3ϕ = 𝛽*I_mask.dot(contagious_vector)
                P3o = (1-σo)*P3ϕ
                P3n = (1-σn)*P3ϕ

            elif t>=3:
                # 進入第三期開始不用管，大家都有口罩
                transition = np.array([[0,0,0],
                                       [0,0,0],
                                       [1,1,1]])
                S_mask = transition.dot(S) # 3x1
                I_mask = transition.dot(I) # 3x1

                matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
                𝛽0 = 𝛽 * matrix
                # 因為 interaction 改變狀態
                dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))


            # moving out from I
            dR = 𝛾 * (1-𝛼) * I_mask
            dD = 𝛾 * 𝛼 * I_mask
            
            # new SIRD
            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            S,I,R,D = nS,nI,nR,nD
            
            t+=1
            
        ##### after for-loop #####
        
        # record probability of being infested
        self.P1ϕ, self.P1o, self.P1n = P1ϕ, P1o, P1n
        self.P2ϕ, self.P2o, self.P2n = P2ϕ, P2o, P2n
        self.P3ϕ, self.P3o, self.P3n = P3ϕ, P3o, P3n
        
        # calculate IC
        phi_sign =(   𝜋B *πB1*( (1-P2n)*(1-P3n) ) +   𝜋B *(1-πB1)*( (1-P2n)*(1-P3o) )+
                   (1-𝜋B)*πB0*( (1-P2ϕ)*(1-P3n) ) +(1-𝜋B)*(1-πB0)*( (1-P2ϕ)*(1-P3ϕ) ) )   
        phi_nsign=(   𝜋A *πA1*( (1-P2n)*(1-P3n) ) +   𝜋A *(1-πA1)*( (1-P2n)*(1-P3o) )+
                   (1-𝜋A)*πA0*( (1-P2ϕ)*(1-P3n) ) +(1-𝜋A)*(1-πA0)*( (1-P2ϕ)*(1-P3ϕ) ) )

        n_sign   =(   𝜋B *𝜋B1*( (1-P2n)*(1-P3n) ) +   𝜋B *(1-𝜋B1)*( (1-P2n)*(1-P3o) )+
                   (1-𝜋B)*𝜋B0*( (1-P2o)*(1-P3n) ) +(1-𝜋B)*(1-𝜋B0)*( (1-P2o)*(1-P3ϕ) ) ) 
        n_nsign  =(   𝜋A *𝜋A1*( (1-P2n)*(1-P3n) ) +   𝜋A *(1-𝜋A1)*( (1-P2n)*(1-P3o) )+ 
                   (1-𝜋A)*𝜋A0*( (1-P2o)*(1-P3n) ) +(1-𝜋A)*(1-𝜋A0)*( (1-P2o)*(1-P3ϕ) ) )
        
        self.ICphi = phi_sign-phi_nsign
        self.ICn   = n_sign-n_nsign 
        
        # record the SIRD
        self.S, self.I, self.R, self.D = sum(S), sum(I), R, D


        
        
def pmix_func(x,m=0.2,mask_supply="benchmark",protect_set="benchmark",I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting parameters
    GRBT=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
    GRBT.S, GRBT.I, GRBT.𝛾, GRBT.𝛼= 1-I, I, 𝛾, 𝛼
    GRBT.p=x
    
    #calculate ICphi
    GRBT.evalD()
    
    return GRBT.ICphi

def qmix_func(x,m=0.2,mask_supply="benchmark",protect_set="benchmark",I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting parameters
    GRBT=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
    GRBT.S, GRBT.I, GRBT.𝛾, GRBT.𝛼= 1-I, I, 𝛾, 𝛼
    GRBT.q=x
    
    #calculate ICn
    GRBT.evalD()
    
    return GRBT.ICn


def solve_GRBT(m=0.2,mask_supply="benchmark",protect_set="benchmark",I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    GRBTp1q0=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
    GRBTp1q0.S, GRBTp1q0.I, GRBTp1q0.𝛾, GRBTp1q0.𝛼= 1-I, I, 𝛾, 𝛼
    GRBTp1q0.evalD()
    
    
    #### Separating Equilibrium ####
    if GRBTp1q0.ICphi>=0 and GRBTp1q0.ICn<=0:
        πB,πA,πA0,πB0,πA1,πB1 = GRBTp1q0.πB, GRBTp1q0.πA, GRBTp1q0.πA0, GRBTp1q0.πB0, GRBTp1q0.πA1, GRBTp1q0.πB1
        ICphi, ICn = GRBTp1q0.ICphi, GRBTp1q0.ICn
        D_val = GRBTp1q0.D
        P2ϕ, P3ϕ = GRBTp1q0.P2ϕ, GRBTp1q0.P3ϕ
        
        return m,1,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,P2ϕ,P3ϕ,D_val
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    elif GRBTp1q0.ICphi<0:
        
        # judge the sign of ICphi
        ICphip1 = pmix_func(1,m,mask_supply=mask_supply,protect_set=protect_set,I=I,𝛾=𝛾,𝛼=𝛼)
        ICphip0 = pmix_func(0,m,mask_supply=mask_supply,protect_set=protect_set,I=I,𝛾=𝛾,𝛼=𝛼)
        
        #### Partial-Separating Equilibrium ####
        # People without mask play mix strategy
        if np.sign(ICphip1)!=np.sign(ICphip0): 
        
            # setting for p=x q=0
            x=bisect(pmix_func, 1e-8, 1, args=(m,mask_supply,protect_set,I,𝛾,𝛼), maxiter=100, full_output=False)
            GRBTpxq0=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
            GRBTpxq0.S, GRBTpxq0.I, GRBTpxq0.𝛾, GRBTpxq0.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTpxq0.p=x

            # calculate ICs and number of death
            GRBTpxq0.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTpxq0.πB, GRBTpxq0.πA, GRBTpxq0.πA0, GRBTpxq0.πB0, GRBTpxq0.πA1, GRBTpxq0.πB1
            ICphi, ICn = GRBTpxq0.ICphi, GRBTpxq0.ICn
            D_val = GRBTpxq0.D
            P2ϕ, P3ϕ = GRBTpxq0.P2ϕ, GRBTpxq0.P3ϕ

            return m,x,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,P2ϕ,P3ϕ,D_val
       
        #### Fully-Pooling Equilibrium ####
        else:
            
            # setting for p=0 q=0
            GRBTp0q0=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
            GRBTp0q0.S, GRBTp0q0.I, GRBTp0q0.𝛾, GRBTp0q0.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTp0q0.p=0
            
            # calculate ICs and number of death
            GRBTp0q0.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTp0q0.πB, GRBTp0q0.πA, GRBTp0q0.πA0, GRBTp0q0.πB0, GRBTp0q0.πA1, GRBTp0q0.πB1
            ICphi, ICn = GRBTp0q0.ICphi, GRBTp0q0.ICn
            D_val = GRBTp0q0.D
            P2ϕ, P3ϕ = GRBTp0q0.P2ϕ, GRBTp0q0.P3ϕ

            return m,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,P2ϕ,P3ϕ,D_val
        
    
    elif GRBTp1q0.ICn>0:
        
        # judge the sign of ICphi
        ICnq1 = qmix_func(1,m,mask_supply=mask_supply,protect_set=protect_set,I=I,𝛾=𝛾,𝛼=𝛼)
        ICnq0 = qmix_func(0,m,mask_supply=mask_supply,protect_set=protect_set,I=I,𝛾=𝛾,𝛼=𝛼)
        
        #### Partial-Separating Equilibrium ####
        # People without mask play mix strategy
        if np.sign(ICnq1)!=np.sign(ICnq0):
            
            x=bisect(qmix_func, 0, 1, args=(m,mask_supply,protect_set,I,𝛾,𝛼), maxiter=100, full_output=False)
            # setting for p=0 q=x
            GRBTp1qx=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
            GRBTp1qx.S, GRBTp1qx.I, GRBTp1qx.𝛾, GRBTp1qx.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTp1qx.q=x

            # calculate ICs and number of death
            GRBTp1qx.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTp1qx.πB, GRBTp1qx.πA, GRBTp1qx.πA0, GRBTp1qx.πB0, GRBTp1qx.πA1, GRBTp1qx.πB1
            ICphi, ICn = GRBTp1qx.ICphi, GRBTp1qx.ICn
            D_val = GRBTp1qx.D
            P2ϕ, P3ϕ = GRBTp1qx.P2ϕ, GRBTp1qx.P3ϕ

            return m,0,x,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,P2ϕ,P3ϕ,D_val
       
        #### Fully-Pooling Equilibrium ####
        else:
            
            # setting for p=1 q=1
            GRBTp1q1=Dynamics(m=m,mech="GRBT",mask_supply=mask_supply,protect_set=protect_set)
            GRBTp1q1.S, GRBTp1q1.I, GRBTp1q1.𝛾, GRBTp1q1.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTp1q1.q=1
            
            # calculate ICs and number of death
            GRBTp1q1.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTp1q1.πB, GRBTp1q1.πA, GRBTp1q1.πA0, GRBTp1q1.πB0, GRBTp1q1.πA1, GRBTp1q1.πB1
            ICphi, ICn = GRBTp1q1.ICphi, GRBTp1q1.ICn
            D_val = GRBTp1q1.D
            P2ϕ, P3ϕ = GRBTp1q1.P2ϕ, GRBTp1q1.P3ϕ

            return m,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,P2ϕ,P3ϕ,D_val

        
def solve_SRA1(m,mask_supply="benchmark",protect_set="benchmark",I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting and calculate parameters
    SRA1=Dynamics(m=m,mech="SRA1",mask_supply=mask_supply,protect_set=protect_set)
    SRA1.S, SRA1.I, SRA1.𝛾, SRA1.𝛼= 1-I, I, 𝛾, 𝛼
    SRA1.evalD()
    
    # calculate probs and number of death
    πB,πA,πA0,πB0,πA1,πB1 = SRA1.πB, SRA1.πA, SRA1.πA0, SRA1.πB0, SRA1.πA1, SRA1.πB1
    D_val = SRA1.D
    
    return m,πB,πA,πA0,πB0,πA1,πB1,D_val

def solve_SRA2(m,mask_supply="benchmark",protect_set="benchmark",I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting and calculate parameters
    SRA2=Dynamics(m=m,mech="SRA2",mask_supply=mask_supply,protect_set=protect_set)
    SRA2.S, SRA2.I, SRA2.𝛾, SRA2.𝛼= 1-I, I, 𝛾, 𝛼
    SRA2.evalD()
    
    # calculate probs and number of death
    πB,πA,πA0,πB0,πA1,πB1 = SRA2.πB, SRA2.πA, SRA2.πA0, SRA2.πB0, SRA2.πA1, SRA2.πB1
    D_val = SRA2.D
    
    return m,πB,πA,πA0,πB0,πA1,πB1,D_val
        
    

    
    
    
    
    
    
    

    

