from gekko import GEKKO
import numpy as np
import csv

'''calculate the probability of reciving a mask in optimal */optimal mechanism'''
def gekko(m0,m1,m2,S0=0.99,δn=0.7,δo=0.5,σn=0.7,σo=0.5,𝛽=2.4/(18/14),𝛾=1-(17/18)**14,v=0.5,ρ=1,α=0.0138,θ=0.2):

    α_star = 2*α            # the mortality rate for those who are quarantined
    I0 = 1-S0               # initial infected
    N0 = 𝛾*α*I0+θ*(1-𝛾)*I0  # number of death plus number of people in quarantine in period 0
    
    m = GEKKO(remote=False)   # create a GEKKO model m
    m.options.OTOL = 1.0e-10  # set 'objective function tolerance for successful solution' to 1.0e-10
    m.options.RTOL = 1.0e-10  # set 'equation solution tolerance' to 1.0e-10

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
    dSn0 = m.Intermediate(𝛽*((1-σn)*Sn0*IΦ0 + (1-δn)*(1-σn)*Sn0*In0)) # people who receive masks but are infected during period 0

    '''the interaction results at the beginning of period 1'''
    SΦ1 = m.Intermediate(SΦ0-dSΦ0)              # people who stay healthy and don't own masks at the beginning of period 1
    Sn1 = m.Intermediate(Sn0-dSn0)              # people who stay healthy and own masks at the beginning of period 1
    IΦ1 = m.Intermediate(IΦ0*(1-𝛾)*(1-θ)+dSΦ0)  # people who are infectious and don't own masks at the beginning of period 1
    In1 = m.Intermediate(In0*(1-𝛾)*(1-θ)+dSn0)  # people who are infectious and own masks at the beginning of period 1
    NA = m.Intermediate(𝛾*α*(In0)+θ*(1-𝛾)*In0)  # number of death plus number of people in quarantine in period 1
    NB = m.Intermediate(𝛾*α*(IΦ0)+θ*(1-𝛾)*IΦ0)  # number of death plus number of people in quarantine in period 1

    '''allocate masks at the beginning of period 1'''
    # (x,y,z)
    # x=S if one stays healthy at the beginning of period 1, x=I otherwise
    # y=Φ if one does not receive masks during period 0, y=n otherwise
    # z=Φ if one does not receive masks during period 1, z=n otherwise
    SΦΦ = m.Intermediate((1-πB)*SΦ1)  # people who are (S,Φ,Φ)
    SnΦ = m.Intermediate((1-πA)*Sn1)  # people who are (S,n,Φ)
    SΦn = m.Intermediate(πB*SΦ1)      # people who are (S,Φ,n)
    Snn = m.Intermediate(πA*Sn1)      # people who are (S,n,n)
    IΦΦ = m.Intermediate((1-πB)*IΦ1)  # people who are (I,Φ,Φ)
    InΦ = m.Intermediate((1-πA)*In1)  # people who are (I,n,Φ)
    IΦn = m.Intermediate(πB*IΦ1)      # people who are (I,Φ,n)
    Inn = m.Intermediate(πA*In1)      # people who are (I,n,n)

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
    SΦΦ2 = m.Intermediate(SΦΦ-dSΦΦ)             # people who are (S,Φ,Φ)
    SnΦ2 = m.Intermediate(SnΦ-dSnΦ)             # people who are (S,n,Φ)
    SΦn2 = m.Intermediate(SΦn-dSΦn)             # people who are (S,Φ,n)
    Snn2 = m.Intermediate(Snn-dSnn)             # people who are (S,n,n)
    IΦΦ2 = m.Intermediate(IΦΦ*(1-𝛾)*(1-θ)+dSΦΦ) # people who are (I,Φ,Φ)
    InΦ2 = m.Intermediate(InΦ*(1-𝛾)*(1-θ)+dSnΦ) # people who are (I,n,Φ)
    IΦn2 = m.Intermediate(IΦn*(1-𝛾)*(1-θ)+dSΦn) # people who are (I,Φ,n)
    Inn2 = m.Intermediate(Inn*(1-𝛾)*(1-θ)+dSnn) # people who are (I,n,n)
    
    '''allocate masks at the beginning of period 2'''
    SΦ2 = m.Intermediate((1-πB0)*SΦΦ2 + (1-πA0)*SnΦ2)              # people who stay healthy and don't own masks at the beginning of period 2
    So2 = m.Intermediate((1-πB1)*SΦn2 + (1-πA1)*Snn2)              # people who stay healthy and own old masks at the beginning of period 2
    Sn2 = m.Intermediate(πB0*SΦΦ2 + πA0*SnΦ2 + πB1*SΦn2 + πA1*Snn2)# people who stay healthy and receive new masks at the beginning of period 2
    IΦ2 = m.Intermediate((1-πB0)*IΦΦ2 + (1-πA0)*InΦ2)              # people who are infectious and don't own masks at the beginning of period 2
    Io2 = m.Intermediate((1-πB1)*IΦn2 + (1-πA1)*Inn2)              # people who are infectious and own old masks at the beginning of period 2
    In2 = m.Intermediate(πB0*IΦΦ2 + πA0*InΦ2 + πB1*IΦn2 + πA1*Inn2)# people who are infectious and receive new masks at the beginning of period 2

    '''interaction during period period 2'''
    dSΦ2 = m.Intermediate(𝛽*(     1*SΦ2*IΦ2 +        (1-δo)*SΦ2*Io2 +        (1-δn)*SΦ2*In2)) # people who don't own masks and are infected during period 2
    dSo2 = m.Intermediate(𝛽*((1-σo)*So2*IΦ2 + (1-σo)*(1-δo)*So2*Io2 + (1-σo)*(1-δn)*So2*In2)) # people who own old masks and are infected during period 2
    dSn2 = m.Intermediate(𝛽*((1-σn)*Sn2*IΦ2 + (1-σn)*(1-δo)*Sn2*Io2 + (1-σn)*(1-δn)*Sn2*In2)) # people who receive new masks but are infected during period 2
    
    '''the interaction results at the beginning of period 3'''
    S0 = m.Intermediate(SΦ2-dSΦ2)             # people who stay healthy and don't own masks at the beginning of period 3
    S1 = m.Intermediate(So2-dSo2)             # people who stay healthy and own old masks at the beginning of period 3
    S2 = m.Intermediate(Sn2-dSn2)             # people who stay healthy and own new masks at the beginning of period 3
    S = m.Intermediate(S0+S1+S2)              # sum of people who stay healthy at the beginning of period 3
    I0 = m.Intermediate(IΦ2*(1-𝛾)*(1-θ)+dSΦ2) # people who are infectious and don't own masks at the beginning of period 3
    I1 = m.Intermediate(Io2*(1-𝛾)*(1-θ)+dSo2) # people who are infectious and own old masks at the beginning of period 3
    I2 = m.Intermediate(In2*(1-𝛾)*(1-θ)+dSn2) # people who are infectious and own new masks at the beginning of period 3
    I = m.Intermediate(I0+I1+I2)              # sum of people who are infectious at the beginning of period 3

    '''calculate the number of healthy and infected people in the next 150 periods'''
    for i in range(150) :
        nS = m.Intermediate(S-𝛽*(1-δn)*(1-σn)*S*I)
        nI = m.Intermediate((1-θ)*(1-𝛾)*I+𝛽*(1-δn)*(1-σn)*S*I)
        S = m.Intermediate(nS)
        I = m.Intermediate(nI)
    
    '''incentive compatibility constraint which need to be ignored when calculating optimal * mechanism'''
    m.Equation(πB*(1+ρ*(πB1+(1-πB1)*v))+(1-πB)*πB0*ρ>=πA*(1+ρ*(πA1+(1-πA1)*v))+(1-πA)*πA0*ρ)         # for people who don't receive masks at t=0
    m.Equation(πA*(1+ρ*(πA1+(1-πA1)*v))+(1-πA)*(v+πA0*ρ)>=πB*(1+ρ*(πB1+(1-πB1)*v))+(1-πB)*(v+πB0*ρ)) # for people who receive masks at t=0

    '''resource constraint'''
    m.Equation( πB*(1-N0)*(1-m0) + πA*(1-N0)*m0 <=m1) # for period 1
    m.Equation( πB*(1-N0)*(1-m0) + πA*(1-N0)*m0 + (πA1*πA+πA0*(1-πA))*(m0*(1-N0)-NA) + (πB1*πB+πB0*(1-πB))*((1-m0)*(1-N0)-NB) <=m1+m2) # for period 1 and 2

    m.Obj((1-S)*( 𝛾*α/(𝛾+(1-𝛾)*θ) + (1-𝛾)*θ*α_star/(𝛾+(1-𝛾)*θ) )) # minimize the number of deaths after 153 periods

    m.solve(disp=False)

    return m.options.objfcnval,πA.value[0],πB.value[0],πA1.value[0],πB1.value[0],πA0.value[0],πB0.value[0],S.value[0],I.value[0]

'''calculate utility levels'''
def Udiff(vo=0.5,vn=0.7,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    
    v=vo/vn  # ratio of utility levels of a new mask over an old mask, vn/vo
  
    Uphi =    𝜌 * vn * (𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌)    # calculate the utility level of those who don't own masks
    Un = vn + 𝜌 * vn * (𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌))# calculate the utility level of those who own masks
    
    return Un, Uphi, Un-Uphi

S0 = 0.99         # initial susceptible
δn,δo = 0.7, 0.5  # new facemask outward protection, old facemask outward protection
σn,σo = 0.7, 0.5  # new facemask inward protection, old facemask inward protection
𝛽 = 2.4/(18/14)   # transmission rate
𝛾 = 1-(17/18)**14 # recovered rate
v = σo/σn         # ratio of utility levels of a new mask over an old mask, vn/vo
ρ = 1             # discount rate
α = 0.0138        # mortality
θ = 0.2           # quarantine power

mask = np.linspace(0.1,0.8,71)
with open ('theta2.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['mask','πA','πB','πA1','πB1','πA0','πB0','S','I','D','Un','Uphi','Un-Uphi'])
    for i in range(len(mask)):
        print(mask[i])
        obj,πA,πB,πA1,πB1,πA0,πB0,S,I = gekko(m0=mask[i],m1=mask[i],m2=mask[i],S0=S0,δn=δn,δo=δo,σn=σn,σo=σo,𝛽=𝛽,𝛾=𝛾,v=v,ρ=ρ,α=α,θ=θ)
        if I>1/1000000:
            print(False)
            break         # If the number of infectious people does not converge to 1/1000000 after 153 periods, stop the simulation.
        Un,Uphi,Un_phi = Udiff(vo=0.5,vn=0.7,𝜌=1,πB=πB,πA=πA,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1) # calculate utility levels
        lst = [πA,πB,πA1,πB1,πA0,πB0,S,I,obj,Un,Uphi,Un_phi]
        lst.insert(0,mask[i])
        writer.writerow(lst)