from gekko import GEKKO

class nation_SIRD :
    def __init__(self,S0=0.99,σn=0.7,σo=0.5,δn=0.7,δo=0.5,𝛽=2.4/(18/14),𝛾=1-(17/18)**14,v=0.5,ρ=1,α=0.0138):

        self.S0 = S0   # initially susceptible
        self.I0 = 1-S0 # initially infected
        self.σn = σn   # new facemask inward protection
        self.σo = σo   # old facemask inward protection
        self.δn = δn   # new facemask outward protection
        self.δo = δo   # old facemask outward protection
        self.𝛽 = 𝛽     # transmission rate
        self.𝛾 = 𝛾     # recovered rate 
        self.v = v     # ratio of utility levels of a new mask over an old mask, vn/vo
        self.ρ = ρ     # discount factor
        self.α = α     # mortality rate

    '''calculate the probability of reciving a mask in optimal mechanism'''
    def find_optimal(self,m0,m1,m2):

        S0,I0 = self.S0, self.I0
        𝛽, 𝛾, α, v, ρ  = self.𝛽, self.𝛾, self.α, self.v, self.ρ
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        D0 = 𝛾*α*I0    # Compute the number of death in period 0

        m = GEKKO(remote=False)  # create a GEKKO model m
        m.options.OTOL = 1.0e-10 # set 'objective function tolerance for successful solution' to 1.0e-10
        m.options.RTOL = 1.0e-10 # set 'equation solution tolerance' to 1.0e-10

        '''declare the variables that need to be calculated, all of the varaibles have the default value 0.5 and upper/lower bound 0/1'''
        πA = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 1 for those who claim they own masks during period 0
        πB = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 1 for those who claim they don't own masks during period 0
        # (x,y) 
        # x=0 if one claims he does not own masks during period 0, x=1 otherwise 
        # y=0 if one does not receive masks during period 1, y=1 otherwise
        πA0 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,0)
        πB0 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,0)
        πA1 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,1)
        πB1 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,1)

        '''allocate masks at the beginning of period 0'''
        SΦ0 = m.Intermediate((1-m0)*S0) # people who stay healthy and don't receive masks at the beginning of period 0
        Sn0 = m.Intermediate(m0*S0)     # people who stay healthy and receive masks at the beginning of period 0
        IΦ0 = m.Intermediate((1-m0)*I0) # people who are infectious and don't receive masks at the beginning of period 0
        In0 = m.Intermediate(m0*I0)     # people who are infectious and receive masks at the beginning of period 0

        '''interaction during period 0'''
        dSΦ0 = m.Intermediate(𝛽*(     1*SΦ0*IΦ0 +        (1-δn)*SΦ0*In0)) # people who don't receive masks and are infected during period 0      
        dSn0 = m.Intermediate(𝛽*((1-σn)*Sn0*IΦ0 + (1-σn)*(1-δn)*Sn0*In0)) # people who receive masks but are infected during period 0

        '''the interaction results at the beginning of period 1'''
        SΦ1 = m.Intermediate(SΦ0-dSΦ0)       # people who stay healthy and don't own masks at the beginning of period 1
        Sn1 = m.Intermediate(Sn0-dSn0)       # people who stay healthy and own masks at the beginning of period 1
        IΦ1 = m.Intermediate(IΦ0*(1-𝛾)+dSΦ0) # people who are infectious and don't own masks at the beginning of period 1
        In1 = m.Intermediate(In0*(1-𝛾)+dSn0) # people who are infectious and own masks at the beginning of period 1
        DA = m.Intermediate(𝛾*α*(In0))       # people who received masks but died at the beginning of period 1  
        DB = m.Intermediate(𝛾*α*(IΦ0))       # people who didn't receive masks and died at the beginning of period 1

        '''allocate masks at the beginning of period 1'''
        # (x,y,z)
        # x=S if one stays healthy at the beginning of period 1, x=I otherwise
        # y=Φ if one does not receive masks during period 0, y=n otherwise
        # z=Φ if one does not receive masks during period 1, z=n otherwise
        SΦΦ = m.Intermediate((1-πB)*SΦ1) # people who are (S,Φ,Φ)
        SnΦ = m.Intermediate((1-πA)*Sn1) # people who are (S,n,Φ)
        SΦn = m.Intermediate(πB*SΦ1)     # people who are (S,Φ,n)
        Snn = m.Intermediate(πA*Sn1)     # people who are (S,n,n)
        IΦΦ = m.Intermediate((1-πB)*IΦ1) # people who are (I,Φ,Φ)
        InΦ = m.Intermediate((1-πA)*In1) # people who are (I,n,Φ)
        IΦn = m.Intermediate(πB*IΦ1)     # people who are (I,Φ,n)
        Inn = m.Intermediate(πA*In1)     # people who are (I,n,n)

        '''interaction during period 1'''
        # (x,y)
        # x=Φ if one does not receive masks during period 0, x=n otherwise
        # y=Φ if one does not receive masks during period 1, y=n otherwise
        dSΦΦ = m.Intermediate(𝛽*(     1*SΦΦ*IΦΦ +        (1-δo)*SΦΦ*InΦ +        (1-δn)*SΦΦ*IΦn +        (1-δn)*SΦΦ*Inn)) # people who are (Φ,Φ) and infected during period 1
        dSnΦ = m.Intermediate(𝛽*((1-σo)*SnΦ*IΦΦ + (1-σo)*(1-δo)*SnΦ*InΦ + (1-σo)*(1-δn)*SnΦ*IΦn + (1-σo)*(1-δn)*SnΦ*Inn)) # people who are (n,Φ) and infected during period 1
        dSΦn = m.Intermediate(𝛽*((1-σn)*SΦn*IΦΦ + (1-σn)*(1-δo)*SΦn*InΦ + (1-σn)*(1-δn)*SΦn*IΦn + (1-σn)*(1-δn)*SΦn*Inn)) # people who are (Φ,n) and infected during period 1
        dSnn = m.Intermediate(𝛽*((1-σn)*Snn*IΦΦ + (1-σn)*(1-δo)*Snn*InΦ + (1-σn)*(1-δn)*Snn*IΦn + (1-σn)*(1-δn)*Snn*Inn)) # people who are (n,n) and infected during period 1

        '''the interaction results at the beginning of period 2'''
        # (x,y,z)
        # x=S if one stays healthy at the beginning of period 2, x=I otherwise
        # y=Φ if one does not receive masks during period 0, y=n otherwise
        # z=Φ if one does not receive masks during period 1, z=n otherwise
        SΦΦ2 = m.Intermediate(SΦΦ-dSΦΦ)         # people who are (S,Φ,Φ) 
        SnΦ2 = m.Intermediate(SnΦ-dSnΦ)         # people who are (S,n,Φ)
        SΦn2 = m.Intermediate(SΦn-dSΦn)         # people who are (S,Φ,n)
        Snn2 = m.Intermediate(Snn-dSnn)         # people who are (S,n,n)
        IΦΦ2 = m.Intermediate(IΦΦ*(1-𝛾)+dSΦΦ)   # people who are (I,Φ,Φ)
        InΦ2 = m.Intermediate(InΦ*(1-𝛾)+dSnΦ)   # people who are (I,n,Φ)
        IΦn2 = m.Intermediate(IΦn*(1-𝛾)+dSΦn)   # people who are (I,Φ,n)
        Inn2 = m.Intermediate(Inn*(1-𝛾)+dSnn)   # people who are (I,n,n)

        '''allocate masks at the beginning of period 2'''
        SΦ2 = m.Intermediate((1-πB0)*SΦΦ2 + (1-πA0)*SnΦ2)               # people who stay healthy and don't own masks at the beginning of period 2
        So2 = m.Intermediate((1-πB1)*SΦn2 + (1-πA1)*Snn2)               # people who stay healthy and own old masks at the beginning of period 2
        Sn2 = m.Intermediate(πB0*SΦΦ2 + πA0*SnΦ2 + πB1*SΦn2 + πA1*Snn2) # people who stay healthy and receive new masks at the beginning of period 2
        IΦ2 = m.Intermediate((1-πB0)*IΦΦ2 + (1-πA0)*InΦ2)               # people who are infectious and don't own masks at the beginning of period 2
        Io2 = m.Intermediate((1-πB1)*IΦn2 + (1-πA1)*Inn2)               # people who are infectious and own old masks at the beginning of period 2
        In2 = m.Intermediate(πB0*IΦΦ2 + πA0*InΦ2 + πB1*IΦn2 + πA1*Inn2) # people who are infectious and receive new masks at the beginning of period 2

        '''interaction during period period 2'''
        dSΦ2 = m.Intermediate(𝛽*(     1*SΦ2*IΦ2 +        (1-δo)*SΦ2*Io2 +        (1-δn)*SΦ2*In2)) # people who don't own masks and are infected during period 2
        dSo2 = m.Intermediate(𝛽*((1-σo)*So2*IΦ2 + (1-σo)*(1-δo)*So2*Io2 + (1-σo)*(1-δn)*So2*In2)) # people who own old masks and are infected during period 2
        dSn2 = m.Intermediate(𝛽*((1-σn)*Sn2*IΦ2 + (1-σn)*(1-δo)*Sn2*Io2 + (1-σn)*(1-δn)*Sn2*In2)) # people who receive new masks but are infected during period 2

        '''the interaction results at the beginning of period 3'''
        S0 = m.Intermediate(SΦ2-dSΦ2)       # people who stay healthy and don't own masks at the beginning of period 3  
        S1 = m.Intermediate(So2-dSo2)       # people who stay healthy and own old masks at the beginning of period 3
        S2 = m.Intermediate(Sn2-dSn2)       # people who stay healthy and own new masks at the beginning of period 3
        S = m.Intermediate(S0+S1+S2)        # sum of people who stay healthy at the beginning of period 3
        I0 = m.Intermediate(IΦ2*(1-𝛾)+dSΦ2) # people who are infectious and don't own masks at the beginning of period 3
        I1 = m.Intermediate(Io2*(1-𝛾)+dSo2) # people who are infectious and own old masks at the beginning of period 3
        I2 = m.Intermediate(In2*(1-𝛾)+dSn2) # people who are infectious and own new masks at the beginning of period 3
        I = m.Intermediate(I0+I1+I2)        # sum of people who are infectious at the beginning of period 3

        '''calculate the number of healthy and infected people in the next 150 periods'''
        for i in range(150) :
            nS = m.Intermediate(S-𝛽*(1-σn)*(1-δn)*S*I)
            nI = m.Intermediate(I+𝛽*(1-σn)*(1-δn)*S*I-𝛾*I)
            S = m.Intermediate(nS)
            I = m.Intermediate(nI)

        '''incentive compatibility constraint'''
        m.Equation(πB*(1+ρ*(πB1+(1-πB1)*v))+(1-πB)*πB0*ρ>=πA*(1+ρ*(πA1+(1-πA1)*v))+(1-πA)*πA0*ρ)        # for people who don't receive masks at t=0
        m.Equation(πA*(1+ρ*(πA1+(1-πA1)*v))+(1-πA)*(v+πA0*ρ)>=πB*(1+ρ*(πB1+(1-πB1)*v))+(1-πB)*(v+πB0*ρ))# for people who receive masks at t=0

        '''resource constraint'''
        m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 <=m1) # for period 1
        m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 + (πA1*πA+πA0*(1-πA))*(m0*(1-D0)-DA) + (πB1*πB+πB0*(1-πB))*((1-m0)*(1-D0)-DB) <=m1+m2) # for period 1 and 2
        
        m.Obj(α*(1-S-I)) # minimize the number of deaths after 153 periods

        m.solve(disp=False)

        return m.options.objfcnval,πA.value[0],πB.value[0],πA1.value[0],πB1.value[0],πA0.value[0],πB0.value[0],S.value[0],I.value[0]

    '''calculate the probability of reciving a mask in optimal * mechanism'''
    def find_optimal_star(self,m0,m1,m2):

        S0,I0 = self.S0, self.I0
        𝛽, 𝛾, α, v, ρ  = self.𝛽, self.𝛾, self.α, self.v, self.ρ
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        D0 = 𝛾*α*I0    # Compute the number of death in period 0

        m = GEKKO(remote=False)  # create a GEKKO model m
        m.options.OTOL = 1.0e-10 # set 'objective function tolerance for successful solution' to 1.0e-10
        m.options.RTOL = 1.0e-10 # set 'equation solution tolerance' to 1.0e-10

        '''declare the variables that need to be calculated, all of the varaibles have the default value 0.5 and upper/lower bound 0/1'''
        πA = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 1 for those who claim they own masks during period 0
        πB = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 1 for those who claim they don't own masks during period 0
        # (x,y) 
        # x=0 if one claims he does not own masks during period 0, x=1 otherwise 
        # y=0 if one does not receive masks during period 1, y=1 otherwise
        πA0 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,0)
        πB0 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,0)
        πA1 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,1)
        πB1 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,1)

        '''allocate masks at the beginning of period 0'''
        SΦ0 = m.Intermediate((1-m0)*S0) # people who stay healthy and don't receive masks at the beginning of period 0
        Sn0 = m.Intermediate(m0*S0)     # people who stay healthy and receive masks at the beginning of period 0
        IΦ0 = m.Intermediate((1-m0)*I0) # people who are infectious and don't receive masks at the beginning of period 0
        In0 = m.Intermediate(m0*I0)     # people who are infectious and receive masks at the beginning of period 0

        '''interaction during period 0'''
        dSΦ0 = m.Intermediate(𝛽*(     1*SΦ0*IΦ0 +        (1-δn)*SΦ0*In0)) # people who don't receive masks and are infected during period 0      
        dSn0 = m.Intermediate(𝛽*((1-σn)*Sn0*IΦ0 + (1-σn)*(1-δn)*Sn0*In0)) # people who receive masks but are infected during period 0

        '''the interaction results at the beginning of period 1'''
        SΦ1 = m.Intermediate(SΦ0-dSΦ0)       # people who stay healthy and don't own masks at the beginning of period 1
        Sn1 = m.Intermediate(Sn0-dSn0)       # people who stay healthy and own masks at the beginning of period 1
        IΦ1 = m.Intermediate(IΦ0*(1-𝛾)+dSΦ0) # people who are infectious and don't own masks at the beginning of period 1
        In1 = m.Intermediate(In0*(1-𝛾)+dSn0) # people who are infectious and own masks at the beginning of period 1
        DA = m.Intermediate(𝛾*α*(In0))       # people who received masks but died at the beginning of period 1  
        DB = m.Intermediate(𝛾*α*(IΦ0))       # people who didn't receive masks and died at the beginning of period 1

        '''allocate masks at the beginning of period 1'''
        # (x,y,z)
        # x=S if one stays healthy at the beginning of period 1, x=I otherwise
        # y=Φ if one does not receive masks during period 0, y=n otherwise
        # z=Φ if one does not receive masks during period 1, z=n otherwise
        SΦΦ = m.Intermediate((1-πB)*SΦ1) # people who are (S,Φ,Φ)
        SnΦ = m.Intermediate((1-πA)*Sn1) # people who are (S,n,Φ)
        SΦn = m.Intermediate(πB*SΦ1)     # people who are (S,Φ,n)
        Snn = m.Intermediate(πA*Sn1)     # people who are (S,n,n)
        IΦΦ = m.Intermediate((1-πB)*IΦ1) # people who are (I,Φ,Φ)
        InΦ = m.Intermediate((1-πA)*In1) # people who are (I,n,Φ)
        IΦn = m.Intermediate(πB*IΦ1)     # people who are (I,Φ,n)
        Inn = m.Intermediate(πA*In1)     # people who are (I,n,n)

        '''interaction during period 1'''
        # (x,y)
        # x=Φ if one does not receive masks during period 0, x=n otherwise
        # y=Φ if one does not receive masks during period 1, y=n otherwise
        dSΦΦ = m.Intermediate(𝛽*(     1*SΦΦ*IΦΦ +        (1-δo)*SΦΦ*InΦ +        (1-δn)*SΦΦ*IΦn +        (1-δn)*SΦΦ*Inn)) # people who are (Φ,Φ) and infected during period 1
        dSnΦ = m.Intermediate(𝛽*((1-σo)*SnΦ*IΦΦ + (1-σo)*(1-δo)*SnΦ*InΦ + (1-σo)*(1-δn)*SnΦ*IΦn + (1-σo)*(1-δn)*SnΦ*Inn)) # people who are (n,Φ) and infected during period 1
        dSΦn = m.Intermediate(𝛽*((1-σn)*SΦn*IΦΦ + (1-σn)*(1-δo)*SΦn*InΦ + (1-σn)*(1-δn)*SΦn*IΦn + (1-σn)*(1-δn)*SΦn*Inn)) # people who are (Φ,n) and infected during period 1
        dSnn = m.Intermediate(𝛽*((1-σn)*Snn*IΦΦ + (1-σn)*(1-δo)*Snn*InΦ + (1-σn)*(1-δn)*Snn*IΦn + (1-σn)*(1-δn)*Snn*Inn)) # people who are (n,n) and infected during period 1

        '''the interaction results at the beginning of period 2'''
        # (x,y,z)
        # x=S if one stays healthy at the beginning of period 2, x=I otherwise
        # y=Φ if one does not receive masks during period 0, y=n otherwise
        # z=Φ if one does not receive masks during period 1, z=n otherwise
        SΦΦ2 = m.Intermediate(SΦΦ-dSΦΦ)         # people who are (S,Φ,Φ) 
        SnΦ2 = m.Intermediate(SnΦ-dSnΦ)         # people who are (S,n,Φ)
        SΦn2 = m.Intermediate(SΦn-dSΦn)         # people who are (S,Φ,n)
        Snn2 = m.Intermediate(Snn-dSnn)         # people who are (S,n,n)
        IΦΦ2 = m.Intermediate(IΦΦ*(1-𝛾)+dSΦΦ)   # people who are (I,Φ,Φ)
        InΦ2 = m.Intermediate(InΦ*(1-𝛾)+dSnΦ)   # people who are (I,n,Φ)
        IΦn2 = m.Intermediate(IΦn*(1-𝛾)+dSΦn)   # people who are (I,Φ,n)
        Inn2 = m.Intermediate(Inn*(1-𝛾)+dSnn)   # people who are (I,n,n)

        '''allocate masks during at the beginning of 2'''
        SΦ2 = m.Intermediate((1-πB0)*SΦΦ2 + (1-πA0)*SnΦ2)               # people who stay healthy and don't own masks at the beginning of period 2
        So2 = m.Intermediate((1-πB1)*SΦn2 + (1-πA1)*Snn2)               # people who stay healthy and own old masks at the beginning of period 2
        Sn2 = m.Intermediate(πB0*SΦΦ2 + πA0*SnΦ2 + πB1*SΦn2 + πA1*Snn2) # people who stay healthy and receive new masks at the beginning of period 2
        IΦ2 = m.Intermediate((1-πB0)*IΦΦ2 + (1-πA0)*InΦ2)               # people who are infectious and don't own masks at the beginning of period 2
        Io2 = m.Intermediate((1-πB1)*IΦn2 + (1-πA1)*Inn2)               # people who are infectious and own old masks at the beginning of period 2
        In2 = m.Intermediate(πB0*IΦΦ2 + πA0*InΦ2 + πB1*IΦn2 + πA1*Inn2) # people who are infectious and receive new masks at the beginning of period 2

        '''interaction during period period 2'''
        dSΦ2 = m.Intermediate(𝛽*(     1*SΦ2*IΦ2 +        (1-δo)*SΦ2*Io2 +        (1-δn)*SΦ2*In2)) # people who don't own masks and are infected during period 2
        dSo2 = m.Intermediate(𝛽*((1-σo)*So2*IΦ2 + (1-σo)*(1-δo)*So2*Io2 + (1-σo)*(1-δn)*So2*In2)) # people who own old masks and are infected during period 2
        dSn2 = m.Intermediate(𝛽*((1-σn)*Sn2*IΦ2 + (1-σn)*(1-δo)*Sn2*Io2 + (1-σn)*(1-δn)*Sn2*In2)) # people who receive new masks but are infected during period 2

        '''the interaction results at the beginning of period 3'''
        S0 = m.Intermediate(SΦ2-dSΦ2)       # people who stay healthy and don't own masks at the beginning of period 3  
        S1 = m.Intermediate(So2-dSo2)       # people who stay healthy and own old masks at the beginning of period 3
        S2 = m.Intermediate(Sn2-dSn2)       # people who stay healthy and own new masks at the beginning of period 3
        S = m.Intermediate(S0+S1+S2)        # sum of people who stay healthy at the beginning of period 3
        I0 = m.Intermediate(IΦ2*(1-𝛾)+dSΦ2) # people who are infectious and don't own masks at the beginning of period 3
        I1 = m.Intermediate(Io2*(1-𝛾)+dSo2) # people who are infectious and own old masks at the beginning of period 3
        I2 = m.Intermediate(In2*(1-𝛾)+dSn2) # people who are infectious and own new masks at the beginning of period 3
        I = m.Intermediate(I0+I1+I2)        # sum of people who are infectious at the beginning of period 3

        '''calculate the number of healthy and infected people in the next 150 periods'''
        for i in range(150) :
            nS = m.Intermediate(S-𝛽*(1-σn)*(1-δn)*S*I)
            nI = m.Intermediate(I+𝛽*(1-σn)*(1-δn)*S*I-𝛾*I)
            S = m.Intermediate(nS)
            I = m.Intermediate(nI)

        '''resource constraint'''
        m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 <=m1) # for period 1
        m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 + (πA1*πA+πA0*(1-πA))*(m0*(1-D0)-DA) + (πB1*πB+πB0*(1-πB))*((1-m0)*(1-D0)-DB) <=m1+m2) # for period 1 and 2
        
        m.Obj(α*(1-S-I)) # minimize the number of deaths after 153 periods

        m.solve(disp=False)

        return m.options.objfcnval,πA.value[0],πB.value[0],πA1.value[0],πB1.value[0],πA0.value[0],πB0.value[0],S.value[0],I.value[0]

'''calculate utility levels'''
def Udiff(vo=0.5,vn=0.7,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    
    v=vo/vn # ratio of utility levels of a new mask over an old mask, vn/vo
    
    Uphi =    𝜌 * vn * (𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌)     # calculate the utility level of those who don't own masks
    Un = vn + 𝜌 * vn * (𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌)) # calculate the utility level of those who own masks
    
    return Un, Uphi, Un-Uphi