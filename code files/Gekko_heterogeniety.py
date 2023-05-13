from gekko import GEKKO
import numpy as np
import csv

'''calculate the probability of reciving a mask in optimal */optimal mechanism'''
def gekko(m0,m1,m2,S0=0.99,δn=0.7,δo=0.5,σn=0.7,σo=0.5,𝛽H=4.8/(18/14),𝛽L=2.4/(18/14),𝛾=1-(17/18)**14,v=0.5,ρ=1,α=0.0138):

    I0 = 1-S0   # initial infected
    D0 = 𝛾*α*I0 # Compute the number of death in period 0

    m = GEKKO(remote=False)  # create a GEKKO model m
    m.options.OTOL = 1.0e-12 # set 'objective function tolerance for successful solution' to 1.0e-12
    m.options.RTOL = 1.0e-12 # set 'equation solution tolerance' to 1.0e-12

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
    SL0 = m.Intermediate((1-m0)*S0) # people who stay healthy and don't receive masks at the beginning of period 0
    SH0 = m.Intermediate(m0*S0)     # people who stay healthy and receive masks at the beginning of period 0
    IL0 = m.Intermediate((1-m0)*I0) # people who are infectious and don't receive masks at the beginning of period 0
    IH0 = m.Intermediate(m0*I0)     # people who are infectious and receive masks at the beginning of period 0

    '''interaction during period 0'''
    dSL0 = m.Intermediate(𝛽L*     1*SL0*IL0 + 𝛽L*       (1-δn)*SL0*IH0) # people who don't receive masks and are infected during period 0            
    dSH0 = m.Intermediate(𝛽H*(1-σn)*SH0*IL0 + 𝛽H*(1-δn)*(1-σn)*SH0*IH0) # people who receive masks but are infected during period 0

    '''the interaction results at the beginning of period 1'''
    SL1 = m.Intermediate(SL0-dSL0)       # people who stay healthy and don't own masks at the beginning of period 1
    SH1 = m.Intermediate(SH0-dSH0)       # people who stay healthy and own masks at the beginning of period 1
    IL1 = m.Intermediate(IL0*(1-𝛾)+dSL0) # people who are infectious and don't own masks at the beginning of period 1
    IH1 = m.Intermediate(IH0*(1-𝛾)+dSH0) # people who are infectious and own masks at the beginning of period 1
    DA = m.Intermediate(𝛾*α*(IH0))       # people who received masks but died at the beginning of period 1
    DB = m.Intermediate(𝛾*α*(IL0))       # people who didn't receive masks and died at the beginning of period 1

    '''allocate masks at the beginning of period 1'''
    # (x,y,z)
    # x=S if one stays healthy at the beginning of period 1, x=I otherwise
    # y=L if one does not receive masks during period 0, y=H otherwise
    # z=Φ if one does not receive masks during period 1, z=n otherwise
    SLΦ1 = m.Intermediate((1-πB)*SL1) # people who are (S,L,Φ)
    SLn1 = m.Intermediate(πB*SL1)     # people who are (S,L,n)
    SHΦ1 = m.Intermediate((1-πA)*SH1) # people who are (S,H,Φ)
    SHn1 = m.Intermediate(πA*SH1)     # people who are (S,H,n)
    ILΦ1 = m.Intermediate((1-πB)*IL1) # people who are (I,L,Φ)
    ILn1 = m.Intermediate(πB*IL1)     # people who are (I,L,n)
    IHΦ1 = m.Intermediate((1-πA)*IH1) # people who are (I,H,Φ)
    IHn1 = m.Intermediate(πA*IH1)     # people who are (I,H,n)

    '''interaction during period 1'''
    # (x,y)
    # x=L if one does not receive masks during period 0, x=H otherwise
    # y=Φ if one does not receive masks during period 1, y=n otherwise
    dSLΦ1 = m.Intermediate(𝛽L*     1*SLΦ1*ILΦ1 + 𝛽L*       (1-δn)*SLΦ1*ILn1 + 𝛽L*       (1-δo)*SLΦ1*IHΦ1 + 𝛽L*       (1-δn)*SLΦ1*IHn1) # people who are (L,Φ) and infected during period 1
    dSLn1 = m.Intermediate(𝛽L*(1-σn)*SLn1*ILΦ1 + 𝛽L*(1-σn)*(1-δn)*SLn1*ILn1 + 𝛽L*(1-σn)*(1-δo)*SLn1*IHΦ1 + 𝛽L*(1-σn)*(1-δn)*SLn1*IHn1) # people who are (L,n) and infected during period 1
    dSHΦ1 = m.Intermediate(𝛽H*(1-σo)*SHΦ1*ILΦ1 + 𝛽H*(1-σo)*(1-δn)*SHΦ1*ILn1 + 𝛽H*(1-σo)*(1-δo)*SHΦ1*IHΦ1 + 𝛽H*(1-σo)*(1-δn)*SHΦ1*IHn1) # people who are (H,Φ) and infected during period 1
    dSHn1 = m.Intermediate(𝛽H*(1-σn)*SHn1*ILΦ1 + 𝛽H*(1-σn)*(1-δn)*SHn1*ILn1 + 𝛽H*(1-σn)*(1-δo)*SHn1*IHΦ1 + 𝛽H*(1-σn)*(1-δn)*SHn1*IHn1) # people who are (H,n) and infected during period 1

    '''the interaction results at the beginning of period 2'''
    # (x,y,z)
    # x=S if one stays healthy at the beginning of period 2, x=I otherwise
    # y=L if one does not receive masks during period 0, y=H otherwise
    # z=Φ if one does not receive masks during period 1, z=n otherwise
    SL_Φ2 = m.Intermediate(SLΦ1-dSLΦ1)       # people who are (S,L,Φ)
    SL_n2 = m.Intermediate(SLn1-dSLn1)       # people who are (S,L,n)
    SH_Φ2 = m.Intermediate(SHΦ1-dSHΦ1)       # people who are (S,H,Φ)
    SH_n2 = m.Intermediate(SHn1-dSHn1)       # people who are (S,H,n)
    IL_Φ2 = m.Intermediate(ILΦ1*(1-𝛾)+dSLΦ1) # people who are (I,L,Φ)
    IL_n2 = m.Intermediate(ILn1*(1-𝛾)+dSLn1) # people who are (I,L,n)
    IH_Φ2 = m.Intermediate(IHΦ1*(1-𝛾)+dSHΦ1) # people who are (I,H,Φ)
    IH_n2 = m.Intermediate(IHn1*(1-𝛾)+dSHn1) # people who are (I,H,n)

    '''allocate masks at the beginning of period 2'''
    SLΦ2 = m.Intermediate((1-πB0)*SL_Φ2)        # people who stay healthy, belong to low-risk group and don't own masks at the beginning of period 2
    SLo2 = m.Intermediate((1-πB1)*SL_n2)        # people who stay healthy, belong to low-risk group and own old masks at the beginning of period 2
    SLn2 = m.Intermediate(πB0*SL_Φ2 + πB1*SL_n2)# people who stay healthy, belong to low-risk group and receive new masks at the beginning of period 2
    SHΦ2 = m.Intermediate((1-πA0)*SH_Φ2)        # people who stay healthy, belong to high-risk group and don't own masks at the beginning of period 2
    SHo2 = m.Intermediate((1-πA1)*SH_n2)        # people who stay healthy, belong to high-risk group and own old masks at the beginning of period 2
    SHn2 = m.Intermediate(πA0*SH_Φ2 + πA1*SH_n2)# people who stay healthy, belong to high-risk group and receive new masks at the beginning of period 2
    ILΦ2 = m.Intermediate((1-πB0)*IL_Φ2)        # people who are infectious, belong to low-risk group and don't own masks at the beginning of period 2
    ILo2 = m.Intermediate((1-πB1)*IL_n2)        # people who are infectious, belong to low-risk group and own old masks at the beginning of period 2
    ILn2 = m.Intermediate(πB0*IL_Φ2 + πB1*IL_n2)# people who are infectious, belong to low-risk group and receove new masks at the beginning of period 2
    IHΦ2 = m.Intermediate((1-πA0)*IH_Φ2)        # people who are infectious, belong to high-risk group and don't own masks at the beginning of period 2
    IHo2 = m.Intermediate((1-πA1)*IH_n2)        # people who are infectious, belong to high-risk group and own old masks at the beginning of period 2
    IHn2 = m.Intermediate(πA0*IH_Φ2 + πA1*IH_n2)# people who are infectious, belong to high-risk group and receive new masks at the beginning of period 2

    '''interaction during period 2'''
    dSLΦ2 = m.Intermediate(𝛽L*(       SLΦ2*ILΦ2 +        (1-δo)*SLΦ2*ILo2 +        (1-δn)*SLΦ2*ILn2 +        SLΦ2*IHΦ2 +        (1-δo)*SLΦ2*IHo2 +        (1-δn)*SLΦ2*IHn2)) # people who belong to low-risk group, don't own masks and are infected during period 2
    dSLo2 = m.Intermediate(𝛽L*((1-σo)*SLo2*ILΦ2 + (1-σo)*(1-δo)*SLo2*ILo2 + (1-σo)*(1-δn)*SLo2*ILn2 + (1-σo)*SLo2*IHΦ2 + (1-σo)*(1-δo)*SLo2*IHo2 + (1-σo)*(1-δn)*SLo2*IHn2)) # people who belong to low-risk group, own old masks and are infected during period 2
    dSLn2 = m.Intermediate(𝛽L*((1-σn)*SLn2*ILΦ2 + (1-σn)*(1-δo)*SLn2*ILo2 + (1-σn)*(1-δn)*SLn2*ILn2 + (1-σn)*SLn2*IHΦ2 + (1-σn)*(1-δo)*SLn2*IHo2 + (1-σn)*(1-δn)*SLn2*IHn2)) # people who belong to low-risk group, receive new masks but are infected during period 2
    dSHΦ2 = m.Intermediate(𝛽H*(       SHΦ2*ILΦ2 +        (1-δo)*SHΦ2*ILo2 +        (1-δn)*SHΦ2*ILn2 +        SHΦ2*IHΦ2 + (       1-δo)*SHΦ2*IHo2 +        (1-δn)*SHΦ2*IHn2)) # people who belong to high-risk group, don't own masks and are infected during period 2
    dSHo2 = m.Intermediate(𝛽H*((1-σo)*SHo2*ILΦ2 + (1-σo)*(1-δo)*SHo2*ILo2 + (1-σo)*(1-δn)*SHo2*ILn2 + (1-σo)*SHo2*IHΦ2 + (1-σo)*(1-δo)*SHo2*IHo2 + (1-σo)*(1-δn)*SHo2*IHn2)) # people who belong to high-risk group, own old masks and are infected during period 2
    dSHn2 = m.Intermediate(𝛽H*((1-σn)*SHn2*ILΦ2 + (1-σn)*(1-δo)*SHn2*ILo2 + (1-σn)*(1-δn)*SHn2*ILn2 + (1-σn)*SHn2*IHΦ2 + (1-σn)*(1-δo)*SHn2*IHo2 + (1-σn)*(1-δn)*SHn2*IHn2)) # people who belong to high-risk group, receive new masks but are infected during period 2

    '''the interaction results at the beginning of period 3'''
    SL_Φ3 = m.Intermediate(SLΦ2-dSLΦ2)      # people who stay healthy, belong to low-risk group and don't own masks at the beginning of period 3 
    SL_o3 = m.Intermediate(SLo2-dSLo2)      # people who stay healthy, belong to low-risk group and own old masks at the beginning of period 3
    SL_n3 = m.Intermediate(SLn2-dSLn2)      # people who stay healthy, belong to low-risk group and own new masks at the beginning of period 3
    SL = m.Intermediate(SL_Φ3+SL_o3+SL_n3)  # sum of people who stay healthy and belong to low-risk group at the beginning of period 3
    SH_Φ3 = m.Intermediate(SHΦ2-dSHΦ2)      # people who stay healthy, belong to high-risk group and don't own masks at the beginning of period 3
    SH_o3 = m.Intermediate(SHo2-dSHo2)      # people who stay healthy, belong to high-risk group and own old masks at the beginning of period 3
    SH_n3 = m.Intermediate(SHn2-dSHn2)      # people who stay healthy, belong to high-risk group and own new masks at the beginning of period 3
    SH = m.Intermediate(SH_Φ3+SH_o3+SH_n3)  # sum of people who stay healthy and belong to high-risk group at the beginning of period 3
    IL_Φ3 = m.Intermediate(ILΦ2*(1-𝛾)+dSLΦ2)# people who are infectious, belong to low-risk group and don't own masks at the beginning of period 3 
    IL_o3 = m.Intermediate(ILo2*(1-𝛾)+dSLo2)# people who are infectious, belong to low-risk group and own old masks at the beginning of period 3
    IL_n3 = m.Intermediate(ILn2*(1-𝛾)+dSLn2)# people who are infectious, belong to low-risk group and own new masks at the beginning of period 3
    IL = m.Intermediate(IL_Φ3+IL_o3+IL_n3)  # sum of people who are infectious and belong to low-risk group at the beginning of period 3
    IH_Φ3 = m.Intermediate(IHΦ2*(1-𝛾)+dSHΦ2)# people who are infectious, belong to high-risk group and don't own masks at the beginning of period 3
    IH_o3 = m.Intermediate(IHo2*(1-𝛾)+dSHo2)# people who are infectious, belong to high-risk group and own old masks at the beginning of period 3
    IH_n3 = m.Intermediate(IHn2*(1-𝛾)+dSHn2)# people who are infectious, belong to high-risk group and own new masks at the beginning of period 3
    IH = m.Intermediate(IH_Φ3+IH_o3+IH_n3)  # sum of people who are infectious and belong to high-risk group at the beginning of period 3

    '''calculate the number of healthy and infected people in the next 150 periods'''
    for i in range(150):
        dSL = m.Intermediate(𝛽L*(1-σn)*(1-δn)*(SL*IL+SL*IH))
        dSH = m.Intermediate(𝛽H*(1-σn)*(1-δn)*(SH*IL+SH*IH))
        nSL = m.Intermediate(SL-dSL)
        nSH = m.Intermediate(SH-dSH)
        nIL = m.Intermediate(IL*(1-𝛾)+dSL)
        nIH = m.Intermediate(IH*(1-𝛾)+dSH)
        SL = m.Intermediate(nSL)
        SH = m.Intermediate(nSH)
        IL = m.Intermediate(nIL)
        IH = m.Intermediate(nIH)

    '''incentive compatibility constraint which need to be ignored when calculating optimal * mechanism'''
    m.Equation(πB*(1+ρ*(πB1+(1-πB1)*v))+(1-πB)*πB0*ρ>=πA*(1+ρ*(πA1+(1-πA1)*v))+(1-πA)*πA0*ρ)        # for people who don't receive masks at t=0
    m.Equation(πA*(1+ρ*(πA1+(1-πA1)*v))+(1-πA)*(v+πA0*ρ)>=πB*(1+ρ*(πB1+(1-πB1)*v))+(1-πB)*(v+πB0*ρ))# for people who receive masks at t=0

    '''resource constraint'''
    m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 <=m1) # for period 1
    m.Equation( πB*(1-D0)*(1-m0) + πA*(1-D0)*m0 + (πA1*πA+πA0*(1-πA))*(m0*(1-D0)-DA) + (πB1*πB+πB0*(1-πB))*((1-m0)*(1-D0)-DB) <=m1+m2) # for period 1 and 2
    
    m.Obj(α*(1-SL-SH-IL-IH)) # minimize the number of deaths after 153 periods

    m.solve(disp=False)

    return m.options.objfcnval,πA.value[0],πB.value[0],πA1.value[0],πB1.value[0],πA0.value[0],πB0.value[0],nSL.value[0],nSH.value[0],nIL.value[0],nIH.value[0]

'''calculate utility levels'''
def Udiff(vo=0.5,vn=0.7,𝜌=1,πB=0.2,πA=0.2,πB0=0.2,πB1=0.2,πA0=0.2,πA1=0.2):
    
    v=vo/vn  # ratio of utility levels of a new mask over an old mask, vn/vo

    Uphi =    𝜌 * vn * (𝜋B*(1+𝜌*(𝜋B1+(1-𝜋B1)*v)) + (1-𝜋B)*𝜋B0*𝜌)    # calculate the utility level of those who don't own masks
    Un = vn + 𝜌 * vn * (𝜋A*(1+𝜌*(𝜋A1+(1-𝜋A1)*v)) + (1-𝜋A)*(v+𝜋A0*𝜌))# calculate the utility level of those who own masks
    
    return Un, Uphi, Un-Uphi

S0 = 0.99            # initial susceptible
δn,δo = 0.7, 0.27    # new facemask outward protection, old facemask outward protection
σn,σo = 0.7, 0.27    # new facemask inward protection, old facemask inward protection
𝛽H = 2.4/(18/14)     # transmission rate among high-risk groups
𝛽L = 0.1*2.4/(18/14) # transmission rate among low-risk groups
𝛾 = 1-(17/18)**14    # recovered rate 
v = 0.5/0.7          # ratio of utility levels of a new mask over an old mask, vn/vo
ρ = 1                # discount factor
α = 0.0138           # mortality rate

mask = np.linspace(0.1,0.8,71)
with open ('hetero_k0.1_old0.27.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['mask','πA','πB','πA1','πB1','πA0','πB0','S','I','D','Un','Uphi','Un-Uphi'])
    for i in range(len(mask)):
        print(mask[i])
        obj,πA,πB,πA1,πB1,πA0,πB0,SL,SH,IL,IH = gekko(m0=mask[i],m1=mask[i],m2=mask[i],S0=S0,δn=δn,δo=δo,σn=σn,σo=σo,𝛽H=𝛽H,𝛽L=𝛽L,𝛾=𝛾,v=v,ρ=ρ,α=α)
        if (IL+IH)>1/1000000:
            print(False)
            break            # If the number of infectious people does not converge to 1/1000000 after 153 periods, stop the simulation.
        Un,Uphi,Un_phi = Udiff(vo=0.5,vn=0.7,𝜌=1,πB=πB,πA=πA,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1) # calculate utility levels
        lst = [πA,πB,πA1,πB1,πA0,πB0,SL+SH,IL+IH,obj,Un,Uphi,Un_phi]
        lst.insert(0,mask[i])
        writer.writerow(lst)