from gekko import GEKKO
import numpy as np
import csv

'''calculate the probability of reciving a mask in optimal */optimal mechanism'''
def nation_SIRD(m0,m1,m2,m3,S0=0.99,δn=0.7,δo=0.5,σn=0.7,σo=0.5,𝛽=2.4/(18/10),𝛾=1-(17/18)**10,v=0.5/0.7,ρ=1,α=0.0138):

    I0 = 1-S0        # initial infected
    D0 = 𝛾*α*I0      # Compute the number of death in period 0
    ρ1, ρ2 = ρ, ρ**2 # discount factor
    
    m = GEKKO(remote=False)  # create a GEKKO model m
    m.options.SOLVER=3
    m.options.OTOL = 1.0e-6 # set 'objective function tolerance for successful solution' to 1.0e-10
    m.options.RTOL = 1.0e-6 # set 'equation solution tolerance' to 1.0e-10

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
    # (x,y,z)
    # x=0 if one claims he does not own masks during period 0, x=1 otherwise
    # y=0 if one does not receive masks during period 1, y=1 otherwise
    # z=0 if one does not receive masks during period 2, z=1 otherwise
    πA00 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,0,0)
    πB00 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,0,0)
    πA01 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,0,1)
    πB01 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,0,1)
    πA10 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,1,0)
    πB10 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,1,0)
    πA11 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (1,1,1)
    πB11 = m.Var(value=0.5,lb=0,ub=1) # the probability of receiving a mask during period 2 for those with masking history (0,1,1)

    '''allocate masks at the beginning of period 0'''
    SΦ0 = m.Intermediate((1-m0)*S0) # people who stay healthy and don't receive masks at the beginning of period 0
    Sn0 = m.Intermediate(m0*S0)     # people who stay healthy and receive masks at the beginning of period 0
    IΦ0 = m.Intermediate((1-m0)*I0) # people who are infectious and don't receive masks at the beginning of period 0
    In0 = m.Intermediate(m0*I0)     # people who are infectious and receive masks at the beginning of period 0

    '''interaction during period 0'''
    dSΦ0 = m.Intermediate(𝛽*(     1*SΦ0*IΦ0 +        (1-δn)*SΦ0*In0)) # people who don't receive masks and are infected during period 0      
    dSn0 = m.Intermediate(𝛽*((1-σn)*Sn0*IΦ0 + (1-δn)*(1-σn)*Sn0*In0)) # people who receive masks but are infected during period 0

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
    SΦΦ1 = m.Intermediate((1-πB)*SΦ1) # people who are (S,Φ,Φ)
    SnΦ1 = m.Intermediate((1-πA)*Sn1) # people who are (S,n,Φ)
    SΦn1 = m.Intermediate(πB*SΦ1)     # people who are (S,Φ,n)
    Snn1 = m.Intermediate(πA*Sn1)     # people who are (S,n,n)
    IΦΦ1 = m.Intermediate((1-πB)*IΦ1) # people who are (I,Φ,Φ)
    InΦ1 = m.Intermediate((1-πA)*In1) # people who are (I,n,Φ)
    IΦn1 = m.Intermediate(πB*IΦ1)     # people who are (I,Φ,n)
    Inn1 = m.Intermediate(πA*In1)     # people who are (I,n,n)

    '''interaction during period 1'''
    # (x,y)
    # x=Φ if one does not receive masks during period 0, x=n otherwise
    # y=Φ if one does not receive masks during period 1, y=n otherwise
    dSΦΦ1 = m.Intermediate(𝛽*(     1*SΦΦ1*IΦΦ1 +        (1-δo)*SΦΦ1*InΦ1 +        (1-δn)*SΦΦ1*IΦn1 +        (1-δn)*SΦΦ1*Inn1)) # people who are (Φ,Φ) and infected during period 1
    dSnΦ1 = m.Intermediate(𝛽*((1-σo)*SnΦ1*IΦΦ1 + (1-σo)*(1-δo)*SnΦ1*InΦ1 + (1-σo)*(1-δn)*SnΦ1*IΦn1 + (1-σo)*(1-δn)*SnΦ1*Inn1)) # people who are (n,Φ) and infected during period 1
    dSΦn1 = m.Intermediate(𝛽*((1-σn)*SΦn1*IΦΦ1 + (1-σn)*(1-δo)*SΦn1*InΦ1 + (1-σn)*(1-δn)*SΦn1*IΦn1 + (1-σn)*(1-δn)*SΦn1*Inn1)) # people who are (Φ,n) and infected during period 1
    dSnn1 = m.Intermediate(𝛽*((1-σn)*Snn1*IΦΦ1 + (1-σn)*(1-δo)*Snn1*InΦ1 + (1-σn)*(1-δn)*Snn1*IΦn1 + (1-σn)*(1-δn)*Snn1*Inn1)) # people who are (n,n) and infected during period 1
    
    P2Φ = m.Intermediate(dSΦΦ1/SΦΦ1)
    P2o = m.Intermediate((1-σo)*P2Φ)
    P2n = m.Intermediate((1-σn)*P2Φ)

    '''the interaction results at the beginning of period 2'''
    # (x,y,z)
    # x=S if one stays healthy at the beginning of period 2, x=I otherwise
    # y=Φ if one does not receive masks during period 0, y=n otherwise
    # z=Φ if one does not receive masks during period 1, z=n otherwise
    SΦΦ2 = m.Intermediate(SΦΦ1-dSΦΦ1)       # people who are (S,Φ,Φ)
    SnΦ2 = m.Intermediate(SnΦ1-dSnΦ1)       # people who are (S,n,Φ)
    SΦn2 = m.Intermediate(SΦn1-dSΦn1)       # people who are (S,Φ,n)
    Snn2 = m.Intermediate(Snn1-dSnn1)       # people who are (S,n,n)
    IΦΦ2 = m.Intermediate(IΦΦ1*(1-𝛾)+dSΦΦ1) # people who are (I,Φ,Φ)
    InΦ2 = m.Intermediate(InΦ1*(1-𝛾)+dSnΦ1) # people who are (I,n,Φ)
    IΦn2 = m.Intermediate(IΦn1*(1-𝛾)+dSΦn1) # people who are (I,Φ,n)
    Inn2 = m.Intermediate(Inn1*(1-𝛾)+dSnn1) # people who are (I,n,n)
    # (x,y)
    # x=0 if one does not receive masks during period 0, x=1 otherwise
    # y=0 if one does not receive masks during period 1, y=1 otherwise
    DA1 = m.Intermediate(𝛾*α*(Inn1)) # people who were (1,1) and died at the beginning of period 2  
    DA0 = m.Intermediate(𝛾*α*(InΦ1)) # people who were (1,0) and died at the beginning of period 2 
    DB1 = m.Intermediate(𝛾*α*(IΦn1)) # people who were (0,1) and died at the beginning of period 2 
    DB0 = m.Intermediate(𝛾*α*(IΦΦ1)) # people who were (0,0) and died at the beginning of period 2 

    '''allocate masks at the beginning of period 2'''
    # (x,y,z,w)
    # x=S if one stays healthy at the beginning of period 2, x=I otherwise
    # y=Φ if one does not receive masks during period 0, y=n otherwise
    # z=Φ if one does not receive masks during period 1, z=n otherwise
    # w=Φ if one does not receive masks during period 2, w=n otherwise
    SΦΦΦ2 = m.Intermediate((1-πB0)*SΦΦ2) # people who are (S,Φ,Φ,Φ)
    SΦΦn2 = m.Intermediate(πB0*SΦΦ2)     # people who are (S,Φ,Φ,n)
    SΦnΦ2 = m.Intermediate((1-πB1)*SΦn2) # people who are (S,Φ,n,Φ)
    SΦnn2 = m.Intermediate(πB1*SΦn2)     # people who are (S,Φ,n,n)
    SnΦΦ2 = m.Intermediate((1-πA0)*SnΦ2) # people who are (S,n,Φ,Φ)
    SnΦn2 = m.Intermediate(πA0*SnΦ2)     # people who are (S,n,Φ,n)
    SnnΦ2 = m.Intermediate((1-πA1)*Snn2) # people who are (S,n,n,Φ)
    Snnn2 = m.Intermediate(πA1*Snn2)     # people who are (S,n,n,n)
    IΦΦΦ2 = m.Intermediate((1-πB0)*IΦΦ2) # people who are (I,Φ,Φ,Φ)
    IΦΦn2 = m.Intermediate(πB0*IΦΦ2)     # people who are (I,Φ,Φ,n)
    IΦnΦ2 = m.Intermediate((1-πB1)*IΦn2) # people who are (I,Φ,n,Φ)
    IΦnn2 = m.Intermediate(πB1*IΦn2)     # people who are (I,Φ,n,n)
    InΦΦ2 = m.Intermediate((1-πA0)*InΦ2) # people who are (I,n,Φ,Φ)
    InΦn2 = m.Intermediate(πA0*InΦ2)     # people who are (I,n,Φ,n)
    InnΦ2 = m.Intermediate((1-πA1)*Inn2) # people who are (I,n,n,Φ)
    Innn2 = m.Intermediate(πA1*Inn2)     # people who are (I,n,n,n)

    '''interaction during period period 2'''
    # (x,y,z)
    # x=Φ if one does not receive masks during period 0, x=n otherwise
    # y=Φ if one does not receive masks during period 1, y=n otherwise
    # z=Φ if one does not receive masks during period 2, z=n otherwise
    dSΦΦΦ2 = m.Intermediate(𝛽*(       SΦΦΦ2*IΦΦΦ2 +        (1-δn)*SΦΦΦ2*IΦΦn2 +        (1-δo)*SΦΦΦ2*IΦnΦ2 +        (1-δn)*SΦΦΦ2*IΦnn2 +        SΦΦΦ2*InΦΦ2 +        (1-δn)*SΦΦΦ2*InΦn2 +        (1-δo)*SΦΦΦ2*InnΦ2 +        (1-δn)*SΦΦΦ2*Innn2)) # people who are (Φ,Φ,Φ) and infected during period 2
    dSΦΦn2 = m.Intermediate(𝛽*((1-σn)*SΦΦn2*IΦΦΦ2 + (1-σn)*(1-δn)*SΦΦn2*IΦΦn2 + (1-σn)*(1-δo)*SΦΦn2*IΦnΦ2 + (1-σn)*(1-δn)*SΦΦn2*IΦnn2 + (1-σn)*SΦΦn2*InΦΦ2 + (1-σn)*(1-δn)*SΦΦn2*InΦn2 + (1-σn)*(1-δo)*SΦΦn2*InnΦ2 + (1-σn)*(1-δn)*SΦΦn2*Innn2)) # people who are (Φ,Φ,n) and infected during period 2
    dSΦnΦ2 = m.Intermediate(𝛽*((1-σo)*SΦnΦ2*IΦΦΦ2 + (1-σo)*(1-δn)*SΦnΦ2*IΦΦn2 + (1-σo)*(1-δo)*SΦnΦ2*IΦnΦ2 + (1-σo)*(1-δn)*SΦnΦ2*IΦnn2 + (1-σo)*SΦnΦ2*InΦΦ2 + (1-σo)*(1-δn)*SΦnΦ2*InΦn2 + (1-σo)*(1-δo)*SΦnΦ2*InnΦ2 + (1-σo)*(1-δn)*SΦnΦ2*Innn2)) # people who are (Φ,n,Φ) and infected during period 2
    dSΦnn2 = m.Intermediate(𝛽*((1-σn)*SΦnn2*IΦΦΦ2 + (1-σn)*(1-δn)*SΦnn2*IΦΦn2 + (1-σn)*(1-δo)*SΦnn2*IΦnΦ2 + (1-σn)*(1-δn)*SΦnn2*IΦnn2 + (1-σn)*SΦnn2*InΦΦ2 + (1-σn)*(1-δn)*SΦnn2*InΦn2 + (1-σn)*(1-δo)*SΦnn2*InnΦ2 + (1-σn)*(1-δn)*SΦnn2*Innn2)) # people who are (Φ,n,n) and infected during period 2
    dSnΦΦ2 = m.Intermediate(𝛽*(       SnΦΦ2*IΦΦΦ2 +        (1-δn)*SnΦΦ2*IΦΦn2 +        (1-δo)*SnΦΦ2*IΦnΦ2 +        (1-δn)*SnΦΦ2*IΦnn2 +        SnΦΦ2*InΦΦ2 +        (1-δn)*SnΦΦ2*InΦn2 +        (1-δo)*SnΦΦ2*InnΦ2 +        (1-δn)*SnΦΦ2*Innn2)) # people who are (n,Φ,Φ) and infected during period 2
    dSnΦn2 = m.Intermediate(𝛽*((1-σn)*SnΦn2*IΦΦΦ2 + (1-σn)*(1-δn)*SnΦn2*IΦΦn2 + (1-σn)*(1-δo)*SnΦn2*IΦnΦ2 + (1-σn)*(1-δn)*SnΦn2*IΦnn2 + (1-σn)*SnΦn2*InΦΦ2 + (1-σn)*(1-δn)*SnΦn2*InΦn2 + (1-σn)*(1-δo)*SnΦn2*InnΦ2 + (1-σn)*(1-δn)*SnΦn2*Innn2)) # people who are (n,Φ,n) and infected during period 2
    dSnnΦ2 = m.Intermediate(𝛽*((1-σo)*SnnΦ2*IΦΦΦ2 + (1-σo)*(1-δn)*SnnΦ2*IΦΦn2 + (1-σo)*(1-δo)*SnnΦ2*IΦnΦ2 + (1-σo)*(1-δn)*SnnΦ2*IΦnn2 + (1-σo)*SnnΦ2*InΦΦ2 + (1-σo)*(1-δn)*SnnΦ2*InΦn2 + (1-σo)*(1-δo)*SnnΦ2*InnΦ2 + (1-σo)*(1-δn)*SnnΦ2*Innn2)) # people who are (n,n,Φ) and infected during period 2
    dSnnn2 = m.Intermediate(𝛽*((1-σn)*Snnn2*IΦΦΦ2 + (1-σn)*(1-δn)*Snnn2*IΦΦn2 + (1-σn)*(1-δo)*Snnn2*IΦnΦ2 + (1-σn)*(1-δn)*Snnn2*IΦnn2 + (1-σn)*Snnn2*InΦΦ2 + (1-σn)*(1-δn)*Snnn2*InΦn2 + (1-σn)*(1-δo)*Snnn2*InnΦ2 + (1-σn)*(1-δn)*Snnn2*Innn2)) # people who are (n,n,n) and infected during period 2
    
    P3Φ = m.Intermediate(dSΦΦΦ2/SΦΦΦ2)
    P3o = m.Intermediate((1-σo)*P3Φ)
    P3n = m.Intermediate((1-σn)*P3Φ)

    '''the interaction results at the beginning of period 3'''
    # (x,y,z,w)
    # x=S if one stays healthy at the beginning of period 3, x=I otherwise
    # y=Φ if one does not receive masks during period 0, y=n otherwise
    # z=Φ if one does not receive masks during period 1, z=n otherwise
    # w=Φ if one does not receive masks during period 2, w=n otherwise
    SΦΦΦ3 = m.Intermediate(SΦΦΦ2-dSΦΦΦ2) # people who are (S,Φ,Φ,Φ)
    SΦΦn3 = m.Intermediate(SΦΦn2-dSΦΦn2) # people who are (S,Φ,Φ,n)
    SΦnΦ3 = m.Intermediate(SΦnΦ2-dSΦnΦ2) # people who are (S,Φ,n,Φ)
    SΦnn3 = m.Intermediate(SΦnn2-dSΦnn2) # people who are (S,Φ,n,n)
    SnΦΦ3 = m.Intermediate(SnΦΦ2-dSnΦΦ2) # people who are (S,n,Φ,Φ)
    SnΦn3 = m.Intermediate(SnΦn2-dSnΦn2) # people who are (S,n,Φ,n)
    SnnΦ3 = m.Intermediate(SnnΦ2-dSnnΦ2) # people who are (S,n,n,Φ)
    Snnn3 = m.Intermediate(Snnn2-dSnnn2) # people who are (S,n,n,n)
    IΦΦΦ3 = m.Intermediate(IΦΦΦ2*(1-𝛾)+dSΦΦΦ2) # people who are (I,Φ,Φ,Φ)
    IΦΦn3 = m.Intermediate(IΦΦn2*(1-𝛾)+dSΦΦn2) # people who are (I,Φ,Φ,n)
    IΦnΦ3 = m.Intermediate(IΦnΦ2*(1-𝛾)+dSΦnΦ2) # people who are (I,Φ,n,Φ)
    IΦnn3 = m.Intermediate(IΦnn2*(1-𝛾)+dSΦnn2) # people who are (I,Φ,n,n)
    InΦΦ3 = m.Intermediate(InΦΦ2*(1-𝛾)+dSnΦΦ2) # people who are (I,n,Φ,Φ)
    InΦn3 = m.Intermediate(InΦn2*(1-𝛾)+dSnΦn2) # people who are (I,n,Φ,n)
    InnΦ3 = m.Intermediate(InnΦ2*(1-𝛾)+dSnnΦ2) # people who are (I,n,n,Φ)
    Innn3 = m.Intermediate(Innn2*(1-𝛾)+dSnnn2) # people who are (I,n,n,n)

    '''allocate masks at the beginning of period 3'''
    SΦ3 = m.Intermediate((1-πB00)*SΦΦΦ3 + (1-πB10)*SΦnΦ3 + (1-πA00)*SnΦΦ3 + (1-πA10)*SnnΦ3)                                   # people who stay healthy and don't own masks at the beginning of period 3
    So3 = m.Intermediate((1-πB01)*SΦΦn3 + (1-πB11)*SΦnn3 + (1-πA01)*SnΦn3 + (1-πA11)*Snnn3)                                   # people who stay healthy and own old masks at the beginning of period 3
    Sn3 = m.Intermediate(πB00*SΦΦΦ3 + πB01*SΦΦn3 + πB10*SΦnΦ3 +πB11*SΦnn3 + πA00*SnΦΦ3 + πA01*SnΦn3 + πA10*SnnΦ3 + πA11*Snnn3)# people who stay healthy and receive new masks at the beginning of period 3
    IΦ3 = m.Intermediate((1-πB00)*IΦΦΦ3 + (1-πB10)*IΦnΦ3 + (1-πA00)*InΦΦ3 + (1-πA10)*InnΦ3)                                   # people who are infectious and don't own masks at the beginning of period 3
    Io3 = m.Intermediate((1-πB01)*IΦΦn3 + (1-πB11)*IΦnn3 + (1-πA01)*InΦn3 + (1-πA11)*Innn3)                                   # people who are infectious and own old masks at the beginning of period 3
    In3 = m.Intermediate(πB00*IΦΦΦ3 + πB01*IΦΦn3 + πB10*IΦnΦ3 +πB11*IΦnn3 + πA00*InΦΦ3 + πA01*InΦn3 + πA10*InnΦ3 + πA11*Innn3)# people who are infectious and receive new masks at the beginning of period 3

    '''interaction during period period 3'''
    dSΦ3 = m.Intermediate(𝛽*(     1*SΦ3*IΦ3 +        (1-δo)*SΦ3*Io3 +        (1-δn)*SΦ3*In3)) # people who don't own masks and are infected during period 3
    dSo3 = m.Intermediate(𝛽*((1-σo)*So3*IΦ3 + (1-σo)*(1-δo)*So3*Io3 + (1-σo)*(1-δn)*So3*In3)) # people who own old masks and are infected during period 3
    dSn3 = m.Intermediate(𝛽*((1-σn)*Sn3*IΦ3 + (1-σn)*(1-δo)*Sn3*Io3 + (1-σn)*(1-δn)*Sn3*In3)) # people who receive new masks but are infected during period 3
    
    P4Φ = m.Intermediate(dSΦ3/SΦ3)
    P4o = m.Intermediate((1-σo)*P4Φ)
    P4n = m.Intermediate((1-σn)*P4Φ)

    '''the interaction results at the beginning of period 4'''
    SΦ4 = m.Intermediate(SΦ3-dSΦ3)       # people who stay healthy and don't own masks at the beginning of period 4
    So4 = m.Intermediate(So3-dSo3)       # people who stay healthy and own old masks at the beginning of period 4
    Sn4 = m.Intermediate(Sn3-dSn3)       # people who stay healthy and own new masks at the beginning of period 4
    IΦ4 = m.Intermediate(IΦ3*(1-𝛾)+dSΦ3) # people who are infectious and don't own masks at the beginning of period 4
    Io4 = m.Intermediate(Io3*(1-𝛾)+dSo3) # people who are infectious and own old masks at the beginning of period 4
    In4 = m.Intermediate(In3*(1-𝛾)+dSn3) # people who are infectious and own new masks at the beginning of period 4

    S = m.Intermediate(SΦ4+So4+Sn4)      # sum of people who stay healthy at the beginning of period 4
    I = m.Intermediate(IΦ4+Io4+In4)      # sum of people who are infectious at the beginning of period 4

    '''calculate the number of healthy and infected people in the next 150 periods'''
    for i in range(300) :
        nS = m.Intermediate(S-𝛽*(1-δn)*(1-σn)*S*I)
        nI = m.Intermediate(I+𝛽*(1-δn)*(1-σn)*S*I-𝛾*I)
        S = m.Intermediate(nS)
        I = m.Intermediate(nI)
    
    '''incentive compatibility constraint which need to be ignored when calculating optimal * mechanism'''                             
    m.Equation(𝜋B*𝜋B1*   𝜋B11 *( (1-P2n)*(1-P3n)*(1-P4n) ) +   (1-𝜋B)*𝜋B0*   𝜋B01 *( (1-P2Φ)*(1-P3n)*(1-P4n) )+
               𝜋B*𝜋B1*(1-𝜋B11)*( (1-P2n)*(1-P3n)*(1-P4o) ) +   (1-𝜋B)*𝜋B0*(1-𝜋B01)*( (1-P2Φ)*(1-P3n)*(1-P4o) )+
               𝜋B*(1-𝜋B1)*𝜋B10*( (1-P2n)*(1-P3o)*(1-P4n) ) +   (1-𝜋B)*(1-𝜋B0)*𝜋B00*( (1-P2Φ)*(1-P3Φ)*(1-P4n) )+
           𝜋B*(1-𝜋B1)*(1-𝜋B10)*( (1-P2n)*(1-P3o)*(1-P4Φ) )+(1-𝜋B)*(1-𝜋B0)*(1-𝜋B00)*( (1-P2Φ)*(1-P3Φ)*(1-P4Φ) )>=
               𝜋A*𝜋A1*   𝜋A11 *( (1-P2n)*(1-P3n)*(1-P4n) ) +   (1-𝜋A)*𝜋A0*   𝜋A01 *( (1-P2Φ)*(1-P3n)*(1-P4n) )+
               𝜋A*𝜋A1*(1-𝜋A11)*( (1-P2n)*(1-P3n)*(1-P4o) ) +   (1-𝜋A)*𝜋A0*(1-𝜋A01)*( (1-P2Φ)*(1-P3n)*(1-P4o) )+
               𝜋A*(1-𝜋A1)*𝜋A10*( (1-P2n)*(1-P3o)*(1-P4n) ) +   (1-𝜋A)*(1-𝜋A0)*𝜋A00*( (1-P2Φ)*(1-P3Φ)*(1-P4n) )+
           𝜋A*(1-𝜋A1)*(1-𝜋A10)*( (1-P2n)*(1-P3o)*(1-P4Φ) )+(1-𝜋A)*(1-𝜋A0)*(1-𝜋A00)*( (1-P2Φ)*(1-P3Φ)*(1-P4Φ) ))  # for people who don't receive masks at t=0
    m.Equation(𝜋A*𝜋A1*   𝜋A11 *( (1-P2n)*(1-P3n)*(1-P4n) ) +   (1-𝜋A)*𝜋A0*   𝜋A01 *( (1-P2o)*(1-P3n)*(1-P4n) )+
               𝜋A*𝜋A1*(1-𝜋A11)*( (1-P2n)*(1-P3n)*(1-P4o) ) +   (1-𝜋A)*𝜋A0*(1-𝜋A01)*( (1-P2o)*(1-P3n)*(1-P4o) )+
               𝜋A*(1-𝜋A1)*𝜋A10*( (1-P2n)*(1-P3o)*(1-P4n) ) +   (1-𝜋A)*(1-𝜋A0)*𝜋A00*( (1-P2o)*(1-P3Φ)*(1-P4n) )+
           𝜋A*(1-𝜋A1)*(1-𝜋A10)*( (1-P2n)*(1-P3o)*(1-P4Φ) )+(1-𝜋A)*(1-𝜋A0)*(1-𝜋A00)*( (1-P2o)*(1-P3Φ)*(1-P4Φ) )>=
               𝜋B*𝜋B1*   𝜋B11 *( (1-P2n)*(1-P3n)*(1-P4n) ) +   (1-𝜋B)*𝜋B0*   𝜋B01 *( (1-P2o)*(1-P3n)*(1-P4n) )+
               𝜋B*𝜋B1*(1-𝜋B11)*( (1-P2n)*(1-P3n)*(1-P4o) ) +   (1-𝜋B)*𝜋B0*(1-𝜋B01)*( (1-P2o)*(1-P3n)*(1-P4o) )+
               𝜋B*(1-𝜋B1)*𝜋B10*( (1-P2n)*(1-P3o)*(1-P4n) ) +   (1-𝜋B)*(1-𝜋B0)*𝜋B00*( (1-P2o)*(1-P3Φ)*(1-P4n) )+
           𝜋B*(1-𝜋B1)*(1-𝜋B10)*( (1-P2n)*(1-P3o)*(1-P4Φ) )+(1-𝜋B)*(1-𝜋B0)*(1-𝜋B00)*( (1-P2o)*(1-P3Φ)*(1-P4Φ) ))# for people who receive masks at t=0
    
    '''resource constraint'''
    m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 <=m1) # for period 1
    m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 + (πA1*πA+πA0*(1-πA))*(m0*(1-D0)-DA) + (πB1*πB+πB0*(1-πB))*((1-m0)*(1-D0)-DB) <=m1+m2) # for period 1 and 2
    m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 + (πA1*πA+πA0*(1-πA))*(m0*(1-D0)-DA) + (πB1*πB+πB0*(1-πB))*((1-m0)*(1-D0)-DB) + (πA11*πA1+πA10*(1-πA1))*(πA*(m0*(1-D0)-DA)-DA1) + (πA01*πA0+πA00*(1-πA0))*((1-πA)*(m0*(1-D0)-DA)-DA0) + (πB11*πB1+πB10*(1-πB1))*(πB*((1-m0)*(1-D0)-DB)-DB1) + (πB01*πB0+πB00*(1-πB0))*((1-πB)*((1-m0)*(1-D0)-DB)-DB0)<=m1+m2+m3) # for period 1, 2 and 3

    m.Obj(α*(1-S-I)) # minimize the number of deaths after 154 periods
    
    try:
        m.solve(disp=False)
    except:
        m.options.SOLVER=2
        m.options.MAX_ITER=5000
        m.solve(disp=False)

    return m.options.objfcnval,πA.value[0],πB.value[0],πA1.value[0],πB1.value[0],πA0.value[0],πB0.value[0],πA11.value[0],πB11.value[0],πA10.value[0],πB10.value[0],πA01.value[0],πB01.value[0],πA00.value[0],πB00.value[0]