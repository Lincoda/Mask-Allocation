import numpy as np
import pandas as pd
from math import log
from scipy.optimize import bisect

class Params():
    def __init__(self,
                 S=0.99,I=0.01,R=0,D=0,       # initial SIRD
                 𝛽=2.4/(18/10),               # dynamic parameters: transmission rate
                 𝛾=1-(17/18)**10,             # dynamic parameters: rate of moving out I
                 𝛼=0.0138,                    # dynamic parameters: death rate
                 
                 σn=0.7,δn=0.7,               # new mask inward/outward protection rate
                 σo=0.5,δo=0.5,               # old mask inward/outward protection rate
                 p=1,q=0,                     # probs of early participation
                 m=0.2,                       # mask supply level
                 mask_supply="benchmark"):    # mask supply pattern                   
        
        self.S, self.I, self.R, self.D = S, I, R, D
        self.𝛽, self.𝛾, self.𝛼 = 𝛽, 𝛾, 𝛼
        self.p, self.q = p, q
        self.σn, self.δn = σn, δn
        self.σo, self.δo = σo, δo
        self.mask_supply = mask_supply
        
    def supply_pattern(self):
        m=self.m
        mask_supply = self.mask_supply
        
        if mask_supply=="benchmark":
            self.m0, self.m1, self.m2, self.m3 = m,m,m,m

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
        
        
        # parameters
        S, I = self.S, self.I
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        m0, m1, m2 = self.m0, self.m1, self.m2
        p, q, = self.p, self.q
        σn,δn = self.σn, self.δn
        
        # numbers of death at the second periods 
        d0 = 𝛾 * 𝛼 * I
        dn = 𝛾 * 𝛼 * ( (1-𝛾)*   m0 *I + 𝛽*   m0 *(1-σn)*S*(m0*(1-δn)*I+(1-m0)*I) )
        dφ = 𝛾 * 𝛼 * ( (1-𝛾)*(1-m0)*I + 𝛽*(1-m0)*       S*(m0*(1-δn)*I+(1-m0)*I) )
        
        # record for later use
        self.d0, self.dn, self.dφ = d0, dn, dφ
        
        ############ GRBT machanism ############
        if self.mech=="GRBT":
            
            # t=1
            πB_coef=   p *(1-m0)*(1-d0) +    q *m0*(1-d0)
            πA_coef=(1-p)*(1-m0)*(1-d0) + (1-q)*m0*(1-d0)
            πB = min(m1/πB_coef,1) if πB_coef>0 else 1
            πA = (m1-πB_coef)/πA_coef if πB==1 else 0

            # t=2 
            if πB<1:
                π2_0_coef=        (1-p)*((1-m0)*(1-d0)-dφ) + (1-q)*(m0*(1-d0)-dn)
                π2_1_coef=(1-πB)*(   p *((1-m0)*(1-d0)-dφ) +    q *(m0*(1-d0)-dn)) 
                π2_2_coef=   πB *(   p *((1-m0)*(1-d0)-dφ) +    q *(m0*(1-d0)-dn))

                πA0 = min(m2/π2_0_coef,1) if π2_0_coef>0 else 1
                πB0 = min((m2-π2_0_coef)/π2_1_coef,1) if πA0==1 else 0
                πA1 = 1 if πB0==1 else 0
                πB1 = min((m2-π2_0_coef-π2_1_coef)/π2_2_coef,1) if πA1==1 else 0

            else:
                π2_0_coef=(1-πA)*((1-p)*((1-m0)*(1-d0)-dφ) + (1-q)*(m0*(1-d0)-dn))
                π2_1_coef=   πA *((1-p)*((1-m0)*(1-d0)-dφ) + (1-q)*(m0*(1-d0)-dn))
                π2_2_coef=           p *((1-m0)*(1-d0)-dφ) +    q *(m0*(1-d0)-dn)


                πA0 = min(m2/π2_0_coef,1) if π2_0_coef>0 else 1
                πB0 = 1 if πA0==1 else 0
                πA1 = min((m2-π2_0_coef)/π2_1_coef,1) if πB0==1 else 0
                πB1 = min((m2-π2_0_coef-π2_1_coef)/π2_2_coef,1) if πA1==1 else 0

            self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1= πB, πA, πA0, πB0, πA1, πB1
        
        ############ SRA-I machanism ############
        elif self.mech=="SRA1":
            
            # t=1
            π1 = m1/(1-d0)
            # t=2 
            π2_coef = (1-m0)*(1-d0)-dφ + m0*(1-d0)-dn
            π2 = m2/π2_coef
            
            πB=πA=π1
            πA0=πB0=πA1=πB1 = π2

            self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1= πB, πA, πA0, πB0, πA1, πB1
        
        ############ SRA-II machanism ############
        elif self.mech=="SRA2":
            
            # t=1
            π1 = m1/(1-d0)
            # t=2 
            π20_coef = (1-π1)*((1-m0)*(1-d0)-dφ + m0*(1-d0)-dn)
            π21_coef =    π1 *((1-m0)*(1-d0)-dφ + m0*(1-d0)-dn)
            
            π20 = min(m2/π20_coef,1)
            π21 = (m2-π20_coef)/π21_coef if π20==1 else 0
            
            πB=πA=π1
            πA0=πB0=π20
            πA1=πB1=π21

            self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1= πB, πA, πA0, πB0, πA1, πB1
            
    def calculate_t3probs(self):
        '''
          t=3 probabilities will depend on t=1 probability. 
          We need to calculate the dynamic to the starting of t=2
        '''
        # calculate the probability in t=1,2
        self.calculate_probs()
        
        # model parameters
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        σn, δn = self.σn, self.δn
        σo, δo = self.σo, self.δo         
        πB, πA, πA0, πB0, πA1, πB1 = self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1
        p, q = self.p, self.q
        m0, m1, m2, m3 = self.m0, self.m1, self.m2, self.m3
        
        
        ##################
        ## SIRD Dynamic ##
        ##################
        
        ##### t=0
        
        S= np.array([(1-m0)*self.S,0,0,0,m0*self.S,0,0,0])
        I= np.array([(1-m0)*self.I,0,0,0,m0*self.I,0,0,0])
        R, D = self.R, self.D
        
        S_mask = S # 8x1
        I_mask = I # 8x1

        # masking state
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
        
        # moving out from I
        dR = 𝛾 * (1-𝛼) * I_mask
        dD = 𝛾 * 𝛼 * I_mask

        # new SIRD
        S1 = S_mask - dS
        I1 = I_mask + dS - dR - dD
        R1 = R + sum(dR)
        D1 = D + sum(dD)
        
        ##### t=1
        
        # early-participation decision
        signup= np.array([[1-p,0,0,0,  0,0,0,0],
                          [  0,0,0,0,  0,0,0,0],
                          [  p,0,0,0,  0,0,0,0],
                          [  0,0,0,0,  0,0,0,0],
                          [  0,0,0,0,1-q,0,0,0],
                          [  0,0,0,0,  0,0,0,0],
                          [  0,0,0,0,  q,0,0,0],
                          [  0,0,0,0,  0,0,0,0]])

        S_signup = signup.dot(S1)
        I_signup = signup.dot(I1)
        
        # getting mask or not
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
        
        # masking state after government issuing masks: φ n φ n o n o n
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
        
        # moving out from I
        dR = 𝛾 * (1-𝛼) * I_mask
        dD = 𝛾 * 𝛼 * I_mask

        # new SIRD
        S2 = S_mask - dS
        I2 = I_mask + dS - dR - dD
        R2 = R1 + sum(dR)
        D2 = D1 + sum(dD)
        
        ##### stop dynamic at t=2
        
        # record the variables of SIRD dynamics
        self.P1ϕ, self.P1o, self.P1n = P1ϕ, P1o, P1n
        self.P2ϕ, self.P2o, self.P2n = P2ϕ, P2o, P2n
        self.S, self.I, self.R, self.D = S2, I2, R2, D2
        
        ######################
        # calculate t3 probs #
        ######################
        
        d0, dn, dφ = self.d0, self.dn, self.dφ
        
        
        # calculate the number of death
        dnB =   q *dn
        dnA =(1-q)*dn
        dφB =   p *dφ
        dφA =(1-p)*dφ
        
        dφA0 = 𝛾 * 𝛼 * ( I2[0] ) # t=3 death of ppl w/o mask choosing A and not getting a mask at t=1
        dφA1 = 𝛾 * 𝛼 * ( I2[1] ) # t=3 death of ppl w/o mask choosing A and getting a mask at t=1
        dφB0 = 𝛾 * 𝛼 * ( I2[2] ) # t=3 death of ppl w/o mask choosing B and not getting a mask at t=1
        dφB1 = 𝛾 * 𝛼 * ( I2[3] ) # t=3 death of ppl w/o mask choosing B and getting a mask at t=1
        
        dnA0 = 𝛾 * 𝛼 * ( I2[4] ) # t=3 death of ppl w/ mask choosing A and not getting a mask at t=1
        dnA1 = 𝛾 * 𝛼 * ( I2[5] ) # t=3 death of ppl w/ mask choosing A and getting a mask at t=1
        dnB0 = 𝛾 * 𝛼 * ( I2[6] ) # t=3 death of ppl w/ mask choosing B and not getting a mask at t=1
        dnB1 = 𝛾 * 𝛼 * ( I2[7] ) # t=3 death of ppl w/ mask choosing B and getting a mask at t=1
        
        ############ GRBT machanism ############
        if self.mech=="GRBT":
            
            πB_coef=   p *(1-m0)*(1-d0) +    q *m0*(1-d0)
            πA_coef=(1-p)*(1-m0)*(1-d0) + (1-q)*m0*(1-d0)
            
            if 𝜋B<1:# A0 B0 B1
                𝜋2_0_coef=          𝜋A_coef - dnA - dφA
                𝜋2_1_coef=(1-𝜋B)*(  𝜋B_coef - dnB - dφB )
                𝜋2_2_coef=   𝜋B *(  𝜋B_coef - dnB - dφB )
                
                if 𝜋A0<1: # A00, B00, A01, B10
                    𝜋3_0_coef= (1-𝜋A0)*( 𝜋2_0_coef-dnA0-dφA0 ) + 𝜋2_1_coef-dnB0-dφB0  #𝜋A00, 𝜋B00
                    𝜋3_1_coef=    𝜋A0 *( 𝜋2_0_coef-dnA0-dφA0 ) + 𝜋2_2_coef-dnB1-dφB1  #𝜋A01, 𝜋B10

                    𝜋A00=min(m3/𝜋3_0_coef,1) if 𝜋3_0_coef>0 else 1
                    𝜋B00=𝜋A00
                    𝜋A01=min((m3-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
                    𝜋A10=𝜋B10=𝜋B01=𝜋A01
                    𝜋B11=0
                    𝜋A11=𝜋B11

                elif 𝜋B0>0 and 𝜋B0<1: # B00, A01, B01, B10
                    𝜋3_0_coef=                        (1-𝜋B0)*( 𝜋2_1_coef-dnB0-dφB0 ) #𝜋B00
                    𝜋3_1_coef= 𝜋2_0_coef-dnA0-dφA0 +     𝜋B0 *( 𝜋2_1_coef-dnB0-dφB0 ) + 𝜋2_2_coef-dnB1-dφB1 #𝜋A01, 𝜋B01, 𝜋B10                         

                    𝜋A00=min(m3/𝜋3_0_coef,1) if 𝜋3_0_coef>0 else 1
                    𝜋B00=𝜋A00
                    𝜋A01=min((m3-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
                    𝜋A10=𝜋B10=𝜋B01=𝜋A01
                    𝜋B11=0
                    𝜋A11=𝜋B11

                elif 𝜋B1>0: # A01 B01 B10 B11
                    𝜋3_1_coef= 𝜋2_0_coef-dnA0-dφA0 + 𝜋2_1_coef-dnB0-dφB0 + (1-𝜋B1)*( 𝜋2_2_coef-dnB1-dφB1 )#𝜋A01, 𝜋B01, 𝜋B10
                    𝜋3_2_coef=                                                𝜋B1 *( 𝜋2_2_coef-dnB1-dφB1 )#𝜋B11                       

                    𝜋A00=1
                    𝜋B00=𝜋A00
                    𝜋A01=min(m3/𝜋3_1_coef,1) if 𝜋3_1_coef>0 else 1
                    𝜋A10=𝜋B10=𝜋B01=𝜋A01
                    𝜋B11=min((m3-𝜋3_1_coef)/𝜋3_2_coef,1) if 𝜋A01==1 else 0
                    𝜋A11=𝜋B11

            else: # A0 A1 B1
                𝜋2_0_coef=(1-𝜋A)*( 𝜋A_coef - dnA - dφA )
                𝜋2_1_coef=   𝜋A *( 𝜋A_coef - dnA - dφA )
                𝜋2_2_coef=         𝜋B_coef - dnB - dφB

                if 𝜋A0<1: # A00 A01 A10 B10
                    𝜋3_0_coef= (1-𝜋A0)*( 𝜋2_0_coef-dnA0-dφA0 ) #𝜋A00
                    𝜋3_1_coef=    𝜋A0 *( 𝜋2_0_coef-dnA0-dφA0 ) + 𝜋2_1_coef-dnA1-dφA1 + 𝜋2_2_coef-dnB1-dφB1 #𝜋A01, #𝜋A10, 𝜋B10

                    𝜋A00=min(m3/𝜋3_0_coef,1) if 𝜋3_0_coef>0 else 1
                    𝜋B00=𝜋A00
                    𝜋A01=min((m3-𝜋3_0_coef)/𝜋3_1_coef,1) if 𝜋A00==1 else 0
                    𝜋A10=𝜋B10=𝜋B01=𝜋A01
                    𝜋B11=0
                    𝜋A11=𝜋B11

                elif 𝜋A1>0 and 𝜋A1<1: # A01 A10 B10 A11
                    𝜋3_1_coef= 𝜋2_0_coef-dnA0-dφA0 + (1-𝜋A1)*( 𝜋2_1_coef-dnA1-dφA1 ) + 𝜋2_2_coef-dnB1-dφB1 #𝜋A01, #𝜋A10 #𝜋B10
                    𝜋3_2_coef=                          𝜋A1 *( 𝜋2_1_coef-dnA1-dφA1 )   #𝜋A11

                    𝜋A00=1
                    𝜋B00=𝜋A00
                    𝜋A01=min(m3/𝜋3_1_coef,1) if 𝜋3_1_coef>0 else 1
                    𝜋A10=𝜋B10=𝜋B01=𝜋A01
                    𝜋B11=min((m3-𝜋3_1_coef)/𝜋3_2_coef,1) if 𝜋A01==1 else 0
                    𝜋A11=𝜋B11

                elif 𝜋B1>0: # A01 A11 B10 B11
                    𝜋3_1_coef= 𝜋2_0_coef-dnA0-dφA0 + (1-𝜋B1)*( 𝜋2_2_coef-dnB1-dφB1 ) #𝜋A01, 𝜋B10
                    𝜋3_2_coef= 𝜋2_1_coef-dnA1-dφA1 +    𝜋B1 *( 𝜋2_2_coef-dnB1-dφB1 ) #𝜋B11, 𝜋A11 

                    𝜋A00=1
                    𝜋B00=𝜋A00
                    𝜋A01=min(m3/𝜋3_1_coef,1) if 𝜋3_1_coef>0 else 1
                    𝜋A10=𝜋B10=𝜋B01=𝜋A01
                    𝜋B11=min((m3-𝜋3_1_coef)/𝜋3_2_coef,1) if 𝜋A01==1 else 0
                    𝜋A11=𝜋B11
            
            # record the probability of getting a mask at t=3
            self.πA00, self.πB00, self.πA01, self.πA10 = πA00, πB00, πA01, πA10 
            self.πB01, self.πB10, self.πA11, self.πB11 = πB01, πB10, πA11, πB11
                    
        ############ SRA-I machanism ############
        elif self.mech=="SRA1":
            
            # t=1
            π1 = m1/(1-d0)
            # t=2 
            π2_coef = (1-m0)*(1-d0)-dφ + m0*(1-d0)-dn
            π2 = m2/π2_coef
            # t=3
            𝜋3_coef = π2_coef -dφA0-dφB0-dnA0-dnB0 -dφA1-dφB1-dnA1-dnB1
            𝜋3 = m3/𝜋3_coef
            
            # record the probability of getting a mask at t=3
            self.πA00=self.πB00=self.πA01=self.πA10=self.πB01=self.πB10=self.πA11=self.πB11=𝜋3
        
        ############ SRA-II machanism ############
        elif self.mech=="SRA2":
            
            # t=1
            π1 = m1/(1-d0)
            # t=2 
            π20_coef = (1-π1)*((1-m0)*(1-d0)-dφ + m0*(1-d0)-dn)
            π21_coef =    π1 *((1-m0)*(1-d0)-dφ + m0*(1-d0)-dn)
            
            π20 = min(m2/π20_coef,1)
            π21 = (m2-π20_coef)/π21_coef if π20==1 else 0
            # t=3
            if 𝜋20 <1:
                𝜋30_coef = (1-𝜋20)*( 𝜋20_coef-dφA0-dφB0-dnA0-dnB0 )
                𝜋31_coef =    𝜋20 *( 𝜋20_coef-dφA0-dφB0-dnA0-dnB0 ) + 𝜋21_coef-dφA1-dφB1-dnA1-dnB1

                𝜋30 = min(m3/𝜋30_coef,1)
                𝜋31 = (m3-𝜋30_coef)/𝜋31_coef if 𝜋30==1 else 0
                𝜋32 = 0


            else:
                𝜋31_coef = 𝜋20_coef-dφA0-dφB0-dnA0-dnB0 + (1-𝜋21)*( 𝜋21_coef-dφA1-dφB1-dnA1-dnB1 )
                𝜋32_coef =                                   𝜋21 *( 𝜋21_coef-dφA1-dφB1-dnA1-dnB1 )    

                𝜋30 = 1
                𝜋31 = min(m3/𝜋31_coef,1)
                𝜋32 = (m3-𝜋31_coef)/𝜋32_coef if 𝜋31==1 else 0
                
            self.𝜋B00=self.𝜋A00=𝜋30
            self.𝜋B10=self.𝜋B01=self.𝜋A01=self.𝜋A10=𝜋31
            self.𝜋A11=self.𝜋B11=𝜋32
            
class Dynamics(GetMaskProbs):
    
    def __init__(self,m,mech):
        
        super().__init__(mech)
        self.m=m
        
        
    def evalD(self):
        
        # calculate the prob of getting a mask
        self.calculate_t3probs()
        
        # model parameters
        S, I, R, D = self.S, self.I, self.R, self.D
        𝛽, 𝛾, 𝛼 = self.𝛽, self.𝛾, self.𝛼
        σn, δn = self.σn, self.δn
        σo, δo = self.σo, self.δo
        πB, πA, πA0, πB0, πA1, πB1 = self.πB, self.πA, self.πA0, self.πB0, self.πA1, self.πB1
        πA00, πB00, πA01, πA10 = self.πA00, self.πB00, self.πA01, self.πA10 
        πB01, πB10, πA11, πB11 = self.πB01, self.πB10, self.πA11, self.πB11
        
        p, q = self.p, self.q
        
        # start the dynamic from t=2
        t=2
        
        for _ in range(2,300):
            
            if t==2:
                
                '''
                 history: φA0 φA1 φB0 φB1 nA0 nA1 nB0 nB1 (get_m0-channel-get_m1 tuple)
                 mask status before issuing mask: ϕ o ϕ o ϕ o ϕ o
                 probs of getting new mask: πA0 πA1 πB0 πB1 πA0 πA1 πB0 πB1
                '''
                transition_2 = np.array([[1-πA0,    0,    0,    0,1-πA0,    0,    0,    0],
                                         [  πA0,    0,    0,    0,  πA0,    0,    0,    0],
                                         [    0,1-πA1,    0,    0,    0,1-πA1,    0,    0],
                                         [    0,  πA1,    0,    0,    0,  πA1,    0,    0],
                                         [    0,    0,1-πB0,    0,    0,    0,1-πB0,    0],
                                         [    0,    0,  πB0,    0,    0,    0,  πB0,    0],
                                         [    0,    0,    0,1-πB1,    0,    0,    0,1-πB1],
                                         [    0,    0,    0,  πB1,    0,    0,    0,  πB1]])
                
                
                S_mask = transition_2.dot(S) # 10x1
                I_mask = transition_2.dot(I) # 10x1
                
                '''
                 Since φ and n going to have different masking state if they are at A10 or B10 at t=3, 
                 so we move these n to 9th and 10th elements.
                 
                 history: φA00+nA00 φA01+nA01 φA10+nA10 φA11+nA11 φB00+nB00 φB01+nB01 φB10+nB10 φB11+nB11   
                 mask status after issuing mask(stock of ϕ/stock of n): ϕ/ϕ n/n o/o n/no ϕ/ϕ n/n o/o n/n
                '''
                contagious_vector = [1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)]
                protective_vector = [1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)]
                matrix = np.outer(contagious_vector,protective_vector)
                𝛽0 = 𝛽 * matrix
                # 因為 interaction 改變狀態
                dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
                # calculate prob of becoming infectious
                P3ϕ = 𝛽*I_mask.dot(contagious_vector)
                P3o = (1-σo)*P3ϕ
                P3n = (1-σn)*P3ϕ
                
                
                
            elif t==3:
                '''
                 history: φA00+nA00 φA01+nA01 φA10+nA10 φA11+nA11 φB00+nB00 φB01+nB01 φB10+nB10 φB11+nB11 
                 mask status before issuing mask(stock of ϕ/stock of n): ϕ/ϕ o/o ϕ/ϕ o/o ϕ/ϕ o/o ϕ/ϕ o/o
                 probs of getting new mask: πA00 πA01 πA10 πA11 πB00 πB01 πB10 πB11
                '''
                transition = np.array([[1-πA00,     0,1-πA10,     0,1-πB00,     0,1-πB10,     0],
                                       [     0,1-πA01,     0,1-πA11,     0,1-πB01,     0,1-πB11],
                                       [  πA00,  πA01,  πA10,  πA11,  πB00,  πB01,  πB10,  πB11]])
                S_mask = transition.dot(S) # 3x1
                I_mask = transition.dot(I) # 3x1
                
                
                # masking state after issuing mask: ϕ o n ϕ o n
                contagious_vector = [1,(1-δo),(1-δn)]
                protective_vector = [1,(1-σo),(1-σn)]
                matrix = np.outer(contagious_vector,protective_vector)
                𝛽0 = 𝛽 * matrix
                # 因為 interaction 改變狀態
                dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
                # calculate prob of becoming infectious
                P4ϕ = 𝛽*I_mask.dot(contagious_vector)
                P4o = (1-σo)*P4ϕ
                P4n = (1-σn)*P4ϕ
                
            elif t>=4:
                # mask supply=1 after T=4
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
            S = S_mask - dS
            I = I_mask + dS - dR - dD
            R = R + sum(dR)
            D = D + sum(dD)
            
            t+=1
            
        ##### after for-loop #####

        # record probability of being infested
        self.P3ϕ, self.P3o, self.P3n = P3ϕ, P3o, P3n
        self.P4ϕ, self.P4o, self.P4n = P4ϕ, P4o, P4n
        
        # use other probability calculated in other function
        P2ϕ, P2o, P2n = self.P2ϕ, self.P2o, self.P2n
        

        # calculate IC

        v=0.5/0.7
        𝜌=1
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
        v_B3 = np.array([1+𝜌+𝜌**2,1+𝜌+𝜌**2*v,1+𝜌*v+𝜌**2,1+𝜌*v,
                         𝜌+𝜌**2,𝜌+𝜌**2*v,𝜌**2,0])
        v_A3 = np.array([0,0,0,0,v,v,v,v])+v_B3

        self.ICphi = P_B3.dot(v_B3) - P_A3.dot(v_B3)
        self.ICn   = P_B3.dot(v_A3) - P_A3.dot(v_A3)

        # record the SIRD
        self.S, self.I, self.R, self.D = sum(S), sum(I), R, D
        
def pmix_func(x,m=0.2,I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting parameters
    GRBT=Dynamics(m=m,mech="GRBT")
    GRBT.S, GRBT.I, GRBT.𝛾, GRBT.𝛼= 1-I, I, 𝛾, 𝛼
    GRBT.p=x
    
    #calculate ICphi
    GRBT.evalD()
    
    return GRBT.ICphi

def qmix_func(x,m=0.2,I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting parameters
    GRBT=Dynamics(m=m,mech="GRBT")
    GRBT.S, GRBT.I, GRBT.𝛾, GRBT.𝛼= 1-I, I, 𝛾, 𝛼
    GRBT.q=x
    
    #calculate ICn
    GRBT.evalD()
    
    return GRBT.ICn
        
        
def solve_SRA1(m,I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting and calculate parameters
    SRA1=Dynamics(m=m,mech="SRA1")
    SRA1.S, SRA1.I, SRA1.𝛾, SRA1.𝛼= 1-I, I, 𝛾, 𝛼
    SRA1.evalD()
    
    # calculate probs and number of death
    π1, π2, π3 = SRA1.πB, SRA1.πB1, SRA1.πB11
    D_val = SRA1.D
    
    return m,π1,π2,π3,D_val

def solve_SRA2(m,I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    # setting and calculate parameters
    SRA2=Dynamics(m=m,mech="SRA2")
    SRA2.S, SRA2.I, SRA2.𝛾, SRA2.𝛼= 1-I, I, 𝛾, 𝛼
    SRA2.evalD()
    
    # calculate probs and number of death
    π1 = SRA2.πB
    π20, π21 = SRA2.πB0, SRA2.πB1
    π30, π31, π32 = SRA2.πB00, SRA2.πB10, SRA2.πB11
    D_val = SRA2.D
    
    return m,π1,π20,π21,π30,π31,π32,D_val
              
      
        
def solve_GRBT(m=0.2,I=0.01,𝛾=1-(17/18)**14,𝛼=0.0138):
    
    GRBTp1q0=Dynamics(m=m,mech="GRBT")
    GRBTp1q0.S, GRBTp1q0.I, GRBTp1q0.𝛾, GRBTp1q0.𝛼= 1-I, I, 𝛾, 𝛼
    GRBTp1q0.evalD()
    
    
    #### Separating Equilibrium ####
    if GRBTp1q0.ICphi>=0 and GRBTp1q0.ICn<=0:
        πB,πA,πA0,πB0,πA1,πB1 = GRBTp1q0.πB, GRBTp1q0.πA, GRBTp1q0.πA0, GRBTp1q0.πB0, GRBTp1q0.πA1, GRBTp1q0.πB1
        ICphi, ICn = GRBTp1q0.ICphi, GRBTp1q0.ICn
        D_val = GRBTp1q0.D
        P2ϕ, P3ϕ, P4ϕ = GRBTp1q0.P2ϕ, GRBTp1q0.P3ϕ, GRBTp1q0.P4ϕ
        
        return m,1,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,D_val
    
    # Individuals whose IC condition does not meet in the fully-separating scenario will deviate.
    # They will deviate by playing a mixed or pure strategy.
    elif GRBTp1q0.ICphi<0:
        
        # judge the sign of ICphi
        ICphip1 = pmix_func(1,m,I=I,𝛾=𝛾,𝛼=𝛼)
        ICphip0 = pmix_func(0,m,I=I,𝛾=𝛾,𝛼=𝛼)
        
        #### Partial-Separating Equilibrium ####
        # People without mask play mix strategy
        if np.sign(ICphip1)!=np.sign(ICphip0): 
        
            # setting for p=x q=0
            x=bisect(pmix_func, 0, 1, args=(m,I,𝛾,𝛼), maxiter=100, full_output=False)
            GRBTpxq0=Dynamics(m=m,mech="GRBT")
            GRBTpxq0.S, GRBTpxq0.I, GRBTpxq0.𝛾, GRBTpxq0.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTpxq0.p=x

            # calculate ICs and number of death
            GRBTpxq0.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTpxq0.πB, GRBTpxq0.πA, GRBTpxq0.πA0, GRBTpxq0.πB0, GRBTpxq0.πA1, GRBTpxq0.πB1
            ICphi, ICn = GRBTpxq0.ICphi, GRBTpxq0.ICn
            D_val = GRBTpxq0.D
            P2ϕ, P3ϕ, P4ϕ = GRBTpxq0.P2ϕ, GRBTpxq0.P3ϕ, GRBTpxq0.P4ϕ

            return m,x,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,D_val
       
        #### Fully-Pooling Equilibrium ####
        else:
            
            # setting for p=0 q=0
            GRBTp0q0=Dynamics(m=m,mech="GRBT")
            GRBTp0q0.S, GRBTp0q0.I, GRBTp0q0.𝛾, GRBTp0q0.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTp0q0.p=0
            
            # calculate ICs and number of death
            GRBTp0q0.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTp0q0.πB, GRBTp0q0.πA, GRBTp0q0.πA0, GRBTp0q0.πB0, GRBTp0q0.πA1, GRBTp0q0.πB1
            ICphi, ICn = GRBTp0q0.ICphi, GRBTp0q0.ICn
            D_val = GRBTp0q0.D
            P2ϕ, P3ϕ = GRBTp0q0.P2ϕ, GRBTp0q0.P3ϕ

            return m,0,0,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,D_val
        
    
    elif GRBTp1q0.ICn>0:
        
        # judge the sign of ICphi
        ICnq1 = qmix_func(1,m,I=I,𝛾=𝛾,𝛼=𝛼)
        ICnq0 = qmix_func(0,m,I=I,𝛾=𝛾,𝛼=𝛼)
        
        #### Partial-Separating Equilibrium ####
        # People without mask play mix strategy
        if np.sign(ICnq1)!=np.sign(ICnq0):
            
            x=bisect(qmix_func, 0, 1, args=(m,I,𝛾,𝛼), maxiter=100, full_output=False)
            # setting for p=0 q=x
            GRBTp1qx=Dynamics(m=m,mech="GRBT")
            GRBTp1qx.S, GRBTp1qx.I, GRBTp1qx.𝛾, GRBTp1qx.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTp1qx.q=x

            # calculate ICs and number of death
            GRBTp1qx.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTp1qx.πB, GRBTp1qx.πA, GRBTp1qx.πA0, GRBTp1qx.πB0, GRBTp1qx.πA1, GRBTp1qx.πB1
            ICphi, ICn = GRBTp1qx.ICphi, GRBTp1qx.ICn
            D_val = GRBTp1qx.D
            P2ϕ, P3ϕ, P4ϕ = GRBTp1qx.P2ϕ, GRBTp1qx.P3ϕ, GRBTp1qx.P4ϕ

            return m,0,x,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,D_val
       
        #### Fully-Pooling Equilibrium ####
        else:
            
            # setting for p=1 q=1
            GRBTp1q1=Dynamics(m=m,mech="GRBT")
            GRBTp1q1.S, GRBTp1q1.I, GRBTp1q1.𝛾, GRBTp1q1.𝛼= 1-I, I, 𝛾, 𝛼
            GRBTp1q1.q=1
            
            # calculate ICs and number of death
            GRBTp1q1.evalD()

            πB,πA,πA0,πB0,πA1,πB1 = GRBTp1q1.πB, GRBTp1q1.πA, GRBTp1q1.πA0, GRBTp1q1.πB0, GRBTp1q1.πA1, GRBTp1q1.πB1
            ICphi, ICn = GRBTp1q1.ICphi, GRBTp1q1.ICn
            D_val = GRBTp1q1.D
            P2ϕ, P3ϕ, P4ϕ = GRBTp1q1.P2ϕ, GRBTp1q1.P3ϕ, GRBTp1q1.P4ϕ

            return m,1,1,ICphi,ICn,πB,πA,πA0,πB0,πA1,πB1,D_val       
    