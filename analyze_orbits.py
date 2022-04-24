'''
This program reads a pickle file containing the time lapse of simulated positions of
asteroids in a 2D plane, limited to interaction with Jupiter and the Sun only.
Simulations are produced by program simulate_orbits.py

Author: Hannes Nilsson
'''

import pickle
import matplotlib.pyplot as plt

def main():

    dict_file = open('2500y_(3_2)_(7_4)_(2_1)_Themis_(13_6)_(9_4)_(7_3)_(29_12)_(5_2)_Ceres_(3_1)_h0.01.pkl', 'rb')
    read_dict = pickle.load(dict_file)

    #DataFrames containing full simulations for each asteroid (equal time)
    a3_2 = read_dict['item0']
    a7_4 = read_dict['item1']
    a2_1 = read_dict['item2']
    aThe = read_dict['item3']
    a7_3 = read_dict['item6']
    a5_2 = read_dict['item8']
    aCer = read_dict['item9']
    a3_1 = read_dict['item10']

    print('Full 2,500 years simulations:\n', a3_2, '\n', a7_4, '\n', a2_1, '\n',
          aThe, '\n', a7_3, '\n', a5_2, '\n', aCer, '\n', a3_1)

    plot_orbit(a3_2, '3:2 2,500 years')
    plot_orbit(a7_4, '7:4 2,500 years')
    plot_orbit(a2_1, '2:1 2,500 years')
    plot_orbit(aThe, 'Themis 2,500 years')
    plot_orbit(a7_3, '7:3 2,500 years')
    plot_orbit(a5_2, '5:2 2,500 years')
    plot_orbit(aCer, 'Ceres 2,500 years')
    plot_orbit(a3_1, '3:1 2,500 years')

    #count the number of orbits for each asteroid in the simulated time,
    #and return the index of the time value when the asteroid reached the
    #maximum number of orbits for the first asteroid (arranged to be the
    #one with longest orbital period.
    o3_2, _ = count_orbits(a3_2,0)
    o3_2, o3_2idx = count_orbits(a3_2,o3_2)
    o7_4, o7_4idx = count_orbits(a7_4,o3_2)
    o2_1, o2_1idx = count_orbits(a2_1,o3_2)
    oThe, oTheidx = count_orbits(aThe,o3_2)
    o7_3, o7_3idx = count_orbits(a7_3,o3_2)
    o5_2, o5_2idx = count_orbits(a5_2,o3_2)
    oCer, oCeridx = count_orbits(aCer,o3_2)
    o3_1, o3_1idx = count_orbits(a3_1,o3_2)

    print('\nNumbers of orbits in 2,500 years for each asteroid: ', o3_2, o2_1, oThe, o7_3, o5_2, oCer, o3_1, '\n') 

    #cut the simulations when reaching the maximum common number of orbits (equal number of orbits)
    a3_2r = limit_orbits(a3_2, o3_2idx)
    a7_4r = limit_orbits(a7_4, o7_4idx)
    a2_1r = limit_orbits(a2_1, o2_1idx)
    aTher = limit_orbits(aThe, oTheidx)
    a7_3r = limit_orbits(a7_3, o7_3idx)
    a5_2r = limit_orbits(a5_2, o5_2idx)
    aCerr = limit_orbits(aCer, oCeridx)
    a3_1r = limit_orbits(a3_1, o3_1idx)

    print('\nReduced simulations limited to ', o3_2, ' orbits:\n', a3_2r, '\n', a7_4r, '\n',
          a2_1r, '\n', aTher, '\n', a7_3r, '\n', a5_2r, '\n', aCerr, '\n', a3_1r)
    
    plot_orbit(a3_2r, '3:2 ' + str(o3_2) + ' orbits')
    plot_orbit(a7_4r, '7:4 ' + str(o3_2) + ' orbits')
    plot_orbit(a2_1r, '2:1 ' + str(o3_2) + ' orbits')
    plot_orbit(aTher, 'Themis ' + str(o3_2) + ' orbits')
    plot_orbit(a7_3r, '7:3 ' + str(o3_2) + ' orbits')
    plot_orbit(a5_2r, '5:2 ' + str(o3_2) + ' orbits')
    plot_orbit(aCerr, 'Ceres ' + str(o3_2) + ' orbits')
    plot_orbit(a3_1r, '3:1 ' + str(o3_2) + ' orbits')

   
#returns how many times an asteroid crosses the positive y-axis
#and what index in the dataframe it completed the nth orbit 
def count_orbits(df,n):
    orbits = 0
    for i in range(df.shape[0]-1):
        if (df['x4=y'].iloc[i] < 1 and df['x4=y'].iloc[i+1] >= 1):
            orbits += 1
        if orbits == n:
            idx = i
    return orbits, idx


#reduces the # of orbits of an asteroid to selected value
def limit_orbits(df, idx):
    return df.iloc[0:idx]
   
def plot_orbit(df, title):
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Initial conditions:\n' + str(df.iloc[0]))
    ax.plot(df['x2=x'], df['x4=y'], df.index, linewidth=0.5)

    ax = fig.add_subplot(1, 3, 3, autoscale_on=False, aspect='equal',
                         xlim=[-4.5,4.5], ylim=[-4.5,4.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Initial conditions:\n' + str(df.iloc[0]))
    ax.plot(df['x2=x'], df['x4=y'], linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    print('Start-up file: analyze_orbits.py')
    main()
