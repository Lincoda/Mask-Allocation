from Gekko_class import *
import numpy as np
import csv

'''
nation = nation_SIRD(parameters)                -> declare a nation with specific parameters
s = 'text.csv'                                  -> the name of the .csv file
mul_0, mul_1, mul_2 = float_1, float_2, float_3 -> multiple of the number of masks 
vo, vn, 𝜌 = float_4, float_5, float_6           -> utility levels of a old mask, utility levels of a new mask, discount factor
'''

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.0138)
# s = 'benchmark_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.35)
# s = 'alpha35_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 1

nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=0.9, v=0.5/0.7, ρ=1, α=0.0138)
s = 'gamma09_star'
mul_0, mul_1, mul_2 = 1,1,1
vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.0138)
# s = 'decreasing_star'
# mul_0, mul_1, mul_2 = 1.15,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.0138)
# s = 'hop_growth_star'
# mul_0, mul_1, mul_2 = 1,1,1.15
# vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.8, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.0138)
# s = 'I2_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.69, δn=0.7, δo=0.69, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.0138)
# s = 'new7_old69_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.69, δn=0.7, δo=0.69, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.69/0.7, ρ=1, α=0.0138)
# s = 'new7_old69_vo_endo_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.69, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.7, σo=0.5, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=0.5, α=0.0138)
# s = 'rho5_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 0.5

# nation = nation_SIRD(S0=0.99, σn=0.6, σo=0.4, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.5/0.7, ρ=1, α=0.0138)
# s = 'sigma64_star'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.5, 0.7, 1

# nation = nation_SIRD(S0=0.99, σn=0.6, σo=0.4, δn=0.7, δo=0.5, 𝛽=2.4/(18/14), 𝛾=1-(17/18)**14, v=0.4/0.6, ρ=1, α=0.0138)
# s = 'sigma64_vo_endo'
# mul_0, mul_1, mul_2 = 1,1,1
# vo, vn, 𝜌 = 0.4, 0.6, 1

mask = np.linspace(0.1,0.8,71)
s = s+'.csv'
'''calculate the probability of reciving a mask in optimal/optimal * mechanism'''
'''start calculating optimal * mechanism if the name of the .csv file contains 'star', otherwise start calculating optimal mechanism'''
if 'star' not in s:
    with open (s,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mask','πA','πB','πA1','πB1','πA0','πB0','S','I','D','Un','Uphi','Un-Uphi'])
        for i in range(len(mask)):
            print(mask[i])
            obj,πA,πB,πA1,πB1,πA0,πB0,S,I = nation.find_optimal(m0=mul_0*mask[i],m1=mul_1*mask[i],m2=mul_2*mask[i])
            if I>1/1000000:
                print(False)
                break        # If the number of infectious people does not converge to 1/1000000 after 153 periods, stop the simulation.
            Un,Uphi,Un_phi = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1) # calculate utility levels
            lst = [πA,πB,πA1,πB1,πA0,πB0,S,I,obj,Un,Uphi,Un_phi]
            lst.insert(0,mask[i])
            writer.writerow(lst)
else:
    with open (s,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mask','πA','πB','πA1','πB1','πA0','πB0','S','I','D','Un','Uphi','Un-Uphi'])
        for i in range(len(mask)):
            print(mask[i])
            obj,πA,πB,πA1,πB1,πA0,πB0,S,I = nation.find_optimal_star(m0=mul_0*mask[i],m1=mul_1*mask[i],m2=mul_2*mask[i])
            if I>1/1000000:
                print(False)
                break        # If the number of infectious people does not converge to 1/1000000 after 153 periods, stop the simulation.
            Un,Uphi,Un_phi = Udiff(vo=vo,vn=vn,𝜌=𝜌,πB=πB,πA=πA,πB0=πB0,πB1=πB1,πA0=πA0,πA1=πA1) # calculate utility levels
            lst = [πA,πB,πA1,πB1,πA0,πB0,S,I,obj,Un,Uphi,Un_phi]
            lst.insert(0,mask[i])
            writer.writerow(lst)
