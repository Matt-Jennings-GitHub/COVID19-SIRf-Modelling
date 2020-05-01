# Matt Jennings
'''
Implementation of Oxford's Susceptible-Infectious-Recovered framework SIRf Coronavirus (CV19) tracking model. Uses SciPy's ODE integrator. Makes use of following coupled ODEs:

dy/dt = by(1-z) - (1/T)*y
dz/dt = by(1-z)
D = Npaz(t-l)
b = R / T


Parameters:
y(t) = proportion of population who are infectious
z(t) = proportion of initial population no longer susceptible to infection (dead, infected, recovered)
D(t) = total deaths
t = time since outbreak began

R = basic reproduction number (2.25 or 2.75 ± 0.025)
T = average infectious period (4.5 ± 1 days)
b = average number of people infected by an infectious individual per day (R/T)
l = average time between infection and death (17 ± 2 days)
a = probability of dying with severe disease (0.14 ± 0.007)
p = proportion of population at risk of severe disease (0.01 or 0.001 ± 50%)
N = size of population (67 million)
'''

#Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

#Functions
# Define a function for integrating the SIRf model

def SIRf(x, t, b, T):
    # Calculates vector [dy/dt,dz/dt] given [y,z,b,T] using the model equations
    y = x[0]
    z = x[1]
    if int(t) < lockdown_time:
        R = R0
    else:
        R = R1
    b2 = R / T
    dydt = b2*y*(1-z) - (1/T)*y
    dzdt = b2*y*(1-z)
    dxdt = [dydt, dzdt]
    return dxdt #return vector of time-derivatives

#Implimentation Parameters
N = 67e6 # inital population size
deltat=0.1 # time-step for the model

t = np.arange(0,365,deltat) # time-step since disease introduced (days)
x0 = [1/N,0] # Initial Condition: [infectious, no longer susceptible] (fraction of population)
R0 =  float(input("Initial R: ")) # veiw effect of lockdown measures
lockdown_time = int(input("Introduce Lockdown on day: "))
R1 = float(input("R after Lockdown: "))

#SIRf Parameters
R = 3 # basic reproduction number # (Make R(t) to simulate impact of lockdown measures)
T = 4.5 # average infectious period 
b = R/T # number of people infected by an infectious individual per day
l = 17 # average time between infection and death 
a = 0.14 # probability of dying with severe disease (Make a(t) to simulate impact of NHS suffering)
p = 0.01 # proportion of population at risk of severe disease

# Integrate the model with SciPi ODE Integrator
xt = odeint(SIRf, x0, t, args=(b, T))
Lambda = np.zeros_like(t)
Lambda[int(l/deltat):] = N*p*a*xt[0:int((365-l)/deltat),1] # Cumulative deaths are a scalar multiple of the number infected, delayed by l

#Display Results
print('Total Deaths:',int(max(Lambda)))
plt.figure(figsize=(14,7))
yscl=['linear','log']

for n in range(0,2):
    plt.subplot(1,2,n+1)
    plt.plot(t, xt[:,0]*N, label='Infectious')
    plt.plot(t, xt[:,1]*N, label='No longer susceptible')
    plt.plot(t, Lambda, label='Total deaths')
    plt.xlabel('Time since outbreak')
    plt.ylabel('Number of people')
    plt.yscale(yscl[n]) #Plot adjacent linear / log graphs
    plt.ylim([1,N])
    plt.legend()
    plt.title('SIRf model predictions for R0 = {}, R1 = {}, Lockdown Day: {}'.format(R0,R1,lockdown_time))


plt.show()
