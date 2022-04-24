'''
This program simulates the orbits of asteroids in the solar system of specified
orbital resonances with Jupiter, intended to study the emergence of Kirkwood gaps.
The model is restricted to 2D motion, and interaction with Jupiter and the Sun only.
Analyzing the outputted simulation is done by the program analyze_orbits.py

Author: Hannes Nilsson
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def main():
    #defining properties of Jupiter and the Sun used in equations of motion for asteroids
    Ms = 1e3
    m2 = 1
    T2 = 11.862  #years
    r2 = 5.2  #A.U.
    M = Ms + m2
    k = np.sqrt((4*np.pi**2*r2**3) / (Ms*T2**2))
    w2 = np.sqrt(k**2*M/r2**3)

    #conditions for simulations
    file_name = '2500y_h0.01.pkl'  #where to save pickle file of simulation
    years = 25  #how long to simulate orbits for
    step_size = 0.01  #to use in RK3 algorithm
    
    #obtain orbital resonance values for asteroids of known orbits (e.g. Ceres)
    '''
    #Ceres
    a1 = 2.7666
    T1 = np.sqrt(a1**3)  #years
    L = T2 / T1
    print('L: ', L)
    w1 = L * w2
    r1 = np.cbrt((2*np.pi / w1)**2)
    '''
    
    dict = {}
    #specify orbital resonance values, calculate initial conditions, and simulate orbits
    L_list = np.array([3/2, 2/1, 2.144, 7.0/3, 5.0/2, 2.576, 3.0/1, 6.739])  #3:2, 2:1, Themis, 7:3, 5:2, Ceres, 3:1, Eros
    for i in range(len(L_list)):
        T1 = T2 / L_list[i]
        w1 = L_list[i] * w2
        r1 = np.cbrt((2*np.pi / w1)**2)
        initial = [0, r1, 0, 0, w1*r1]
        
        vals = rk3(initial,years,step_size)

        dict['item' + str(i)] = vals
    
    f = open(file_name, 'wb')
    pickle.dump(dict, f)
    f.close()

    '''
    #read saved file to analyze simulations
    dict_file = open('file_name', 'rb')
    read_dict = pickle.load(dict_file)

    print(read_dict)

    plot_orbit(vals)
    '''
    
#3rd order runge-kutta method with initial values [t = 0, x(0), x'(0), y(0), y'(0)] stepsize = h
def rk3(initial, years, h):
    x1, x2, x3, x4, x5 = initial
    t = 0
    h = 0.01
    df = pd.DataFrame([initial], columns = ['x1=t', 'x2=x(t)', 'x3=x\'(t)', 'x4=y(t)', 'x5=y\'(t)'])
    for i in range(int(years/h)):
        t = t + h
        a1 = h  #f1 = 1
        b1 = h*x3  #f2 = dx
        d1 = h*x5  #f4 = dy
        c1, e1 = f3f5(x2, x4, t)
        c1 *= h
        e1 *= h
        a1_ = x1 + a1/2
        b1_ = x2 + b1/2
        c1_ = x3 + c1/2
        d1_ = x4 + d1/2
        e1_ = x5 + e1/2

        a2 = h  #f1 = 1
        b2 = h*c1_  #f2 = dx
        d2 = h*e1_  #f4 = dy
        c2, e2 = f3f5(b1_, d1_, t)
        c2 *= h
        e2 *= h
        a2_ = x1 + 2*a2 - a1
        b2_ = x2 + 2*b2 - b1
        c2_ = x3 + 2*c2 - c1
        d2_ = x4 + 2*d2 - d1
        e2_ = x5 + 2*e2 - e1

        a3 = h  #f1 = 1
        b3 = h*c2_  #f2 = dx
        d3 = h*e2_  #f4 = dy
        c3, e3 = f3f5(b2_, d2_, t)
        c3 *= h
        e3 *= h

        x1 += (a1 + 4*a2 + a3)/6
        x2 += (b1 + 4*b2 + b3)/6
        x3 += (c1 + 4*c2 + c3)/6
        x4 += (d1 + 4*d2 + d3)/6
        x5 += (e1 + 4*e2 + e3)/6

        vals = [x1, x2, x3, x4, x5]
        series = pd.Series(vals, index = df.columns)
        df = df.append(series, ignore_index=True)

    return df


def f3f5(x1, y1, t):
    Ms = 1e3
    m2 = 1
    T2 = 11.862  #years
    r2 = 5.2  #A.U.
    M = Ms + m2
    k = np.sqrt((4*np.pi**2*r2**3) / (Ms*T2**2))
    w2 = np.sqrt(k**2*M/r2**3)

    x2 = r2*np.cos(w2*t)
    y2 = r2*np.sin(w2*t)
    r1 = np.sqrt(x1**2 + y1**2)
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    ddx1 = k**2*m2*((x2-x1)/r12**3 - x2/r2**3) - k**2*Ms*x1/r1**3
    ddy1 = k**2*m2*((y2-y1)/r12**3 - y2/r2**3) - k**2*Ms*y1/r1**3
    
    return ddx1, ddy1

def plot_orbit(df):
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Initial conditions:\n' + str(df.iloc[0]))
    ax.plot(df['x2=x(t)'], df['x4=y(t)'], df.index, linewidth=0.5)

    ax = fig.add_subplot(1, 3, 3, autoscale_on=False, aspect='equal',
                         xlim=[-4.2,4.2], ylim=[-4.2,4.2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Initial conditions:\n' + str(df.iloc[0]))
    ax.plot(df['x2=x(t)'], df['x4=y(t)'], linewidth=0.5)
    plt.show()
    
if __name__ == '__main__':
    print('Start-up file: simulate_orbits.py')
    main()
