#Particle swarm optimization
from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def depthcalc(x,y):
    """Objective Function"""
    """Test functions"""
    """1 - Rastrign Function"""
    # return 20+(x-2)**2 +(y-2)**2 - 10*(cos(2*pi*(x-2))+cos(2*pi*(y-2)))

    """Harmony function"""
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + sin(3 * x + 1.41) + sin(4 * y - 1.73)

    """ Himmelblau's function"""
    # return (x**2+y-11)**2 + (x+y**2 - 7)**2

    """Styblinski-Tang function"""
    # return 0.5 * ((x ** 4 - 16 * x ** 2 + 5 * x) + (y ** 4 - 16 * y ** 2 + 5 * y))

    """Holder table function"""
    # return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))

    """Ackley's function"""
    # return -20*np.exp(-0.2*np.sqrt(0.5*((x-2)**2 + (y-2)**2))) - np.exp(0.5*(np.cos(2*pi*(x-2)) + np.cos(2*pi*(y-2)))) + 20 + np.exp(1)

    """Goldstein Price function"""
    # return (1 + ((x + y + 1)**2) * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + ((2*x - 3*y)**2) * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))


    """Bukin function N.6 """
    # return  100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

def pso(n,w,c1,c2,it,x_max,x_min,y_max,y_min,v_max):
    rows = 2
    cols =5
    fig, ax = plt.subplots(rows, cols, figsize=(15, 6), gridspec_kw={'width_ratios': [1] * cols, 'height_ratios': [1] * rows})
    x, y = meshgrid(linspace(-5, 5, 500), linspace(-5, 5, 500))
    z = depthcalc(x, y)
    # print("amin z",np.amin(z))
    position=np.where((np.round(z,2) == np.amin(np.round(z,2))))
    print("Position",position)
    Best_x = x[position]
    Best_y = y[position]
    d_min = np.amin(z)
    x, y = meshgrid(linspace(-5, 5, 500), linspace(-5, 5, 500))
    z = depthcalc(x, y)
    # Create a grid of particle positions
    # Calculate the grid spacing
    x_spacing = (x_max - x_min) / (int(n** 0.5) - 1)
    y_spacing = (y_max - y_min) / (int(n** 0.5) - 1)

    # Create a grid of particle positions
    X = []
    for i in range(int(n ** 0.5)):
        for j in range(int(n ** 0.5)):
            x1 = x_min + i * x_spacing
            y1 = y_min + j * y_spacing
            X.append([x1, y1])
    r_particles = []
    for i in range(n-(int(np.sqrt(n))**2)):
        x2 = np.random.uniform(x_min, x_max)
        y2 = np.random.uniform(y_min, y_max)
        X.append([x2,y2])


    print(X)


    V = [[np.random.uniform(0, 0.25) for _ in range(2)] for _ in range(n)]  # Initial velocity of the particles in m/s

    # print(X[0][1])
    r = 0
    c = 0
    gmin= 100000000
    pmin = 100000000
    Pbest= [];
    Gbest=[];
    for i in range(it):
        # w = 2 * (1- i / it)
        # c1 = 2 - i/it
        # c2 =  (i*2) / (it)


        r1=random.random()
        r2=random.random()
        # r1=0.5
        # r2=0.5
        div = it/10
        if (i+1)%div == 0:
            ax[r, c].contourf(x, y, z)
            ax[r, c].scatter(Best_x, Best_y, marker='*', color='green')
            ax[r, c].set_title(f"Iteration {i+1}")
            ax[r, c].set_aspect('equal')

        for j in range(n):
            depth = depthcalc(X[j][0], X[j][1])
            # print(depth)
            if depth < pmin:
                Pbest = [X[j][0], X[j][1]]
                pmin = depth
            if depth < gmin:
                Gbest = [X[j][0], X[j][1]]
                gmin=depth

            print("pbest", Pbest)
            print("gbest", Gbest)
            if (i+1)%div==0:
                ax[r, c].scatter(X[j][0], X[j][1], marker='o', color='black')
                ax[r, c].quiver(X[j][0], X[j][1], V[j][0], V[j][1], color='red', angles='xy', scale_units='xy')

        if (i+1)%div ==0:

            if c == 4:
                r = r + 1
                c = -1
            c = c + 1

        v_max = 0.25

        for j in range(n):
            V[j][0] = w * V[j][0] + c1 * r1 * (Pbest[0] - X[j][0]) + c2 * r2 * (Gbest[0] - X[j][0])  # X coordinate velocities
            V[j][1] = w * V[j][1] + c1 * r1 * (Pbest[1] - X[j][1]) + c2 * r2 * (Gbest[1] - X[j][1])  # Y coordinate velocities


            velocity_magnitude = ((V[j][0])**2+(V[j][1])**2)**0.5
            # print(velocity_magnitude)
            if velocity_magnitude > v_max:
                # print(V[j])
                V[j][0] = V[j][0] * (v_max / velocity_magnitude)
                V[j][1]= V[j][1]* (v_max / velocity_magnitude)

            if ((X[j][0] + V[j][0]) < -5) and ((X[j][1] + V[j][1]) > -5) and ((X[j][1] + V[j][1]) < 5):
                V[j][0] = -V[j][0]    #Reflective boundary # Left side

            if ((X[j][0] + V[j][0]) > 5) and ((X[j][1] + V[j][1]) > -5) and ((X[j][1] + V[j][1]) < 5):
                V[j][0] = -V[j][0]    #Reflective boundary Right side

            if ((X[j][1] + V[j][1]) > 5) and ((X[j][0] + V[j][0]) > -5) and ((X[j][0] + V[j][0]) < 5):
                V[j][1] = -V[j][1]    #Reflective boundary Top

            if ((X[j][1] + V[j][1]) < -5) and ((X[j][0] + V[j][0]) > -5) and ((X[j][0] + V[j][0]) < 5):
                V[j][1] = -V[j][1]  # Reflective boundary Bottom

            if (((X[j][0] + V[j][0]) < -5) and ((X[j][1] + V[j][1]) > 5))  or (((X[j][0] + V[j][0]) > 5) and ((X[j][1] + V[j][1]) > 5)) or (((X[j][0] + V[j][0])) > 5 and ((X[j][1] + V[j][1]) < -5))  or (((X[j][0] + V[j][0]) < -5) and ((X[j][1] + V[j][1]) < -5)):
                X[j][0] = round(X[j][0])*0.99
                X[j][1] = round(X[j][1])*0.99
                # print(X[j][0],X[j][1])
                print("Corner condition is active")
                break

            X[j][0] = X[j][0] + V[j][0]  # X coordinate position
            X[j][1] = X[j][1] + V[j][1]  # Y coordinate position
            # print(V)

        print("Iteration", i)
        # print(V)

    print("Analytical maximum depth",d_min)
    print("PSO maximum obtained depth", depthcalc(Gbest[0], Gbest[1]))
    print([Best_x[0],Best_y[0]])
    depth_error = d_min - depthcalc(Gbest[0], Gbest[1])
    loc_error_x = 100
    loc_error_y = 100
    print(len(Best_x))
    for i in range(len(Best_x)):
        if loc_error_x > np.abs(Gbest[0] - (Best_x[i])):
            loc_error_x = np.abs(Gbest[0] - (Best_x[i]))
        if loc_error_y > np.abs(Gbest[1] - (Best_y[i])):
            loc_error_y = np.abs(Gbest[1] - (Best_y[i]))


    loc_error_x = np.abs(Gbest[0] - (Best_x))
    loc_error_y = np.abs(Gbest[1] - (Best_y))


    print("Location error X",loc_error_x,"Location error Y",loc_error_y)
    plt.tight_layout()
    plt.show()
    error= np.abs(depth_error)/np.abs(d_min)*100
    print("Depth error", error)
    errors=[error, loc_error_x, loc_error_y]
    return errors


if __name__ == '__main__':

    #Parameters
    """Number of particles"""
    n =10
    """Inertia weight constant"""
    w = 1
    """Cognitive coefficient"""
    c1=2
    """Social coefficient"""
    c2=2

    """No of iterations"""
    it=50


    #Search space setup and particle limitations
    "x_max , x_min and y_max y_min represent the bounds of the search space"
    "In our search domain 10 units is considered as 1km. Eg: Travelling along x from -5 to +5 is equivalent to 1km"
    x_max = 5
    x_min = -5
    y_max = 5
    y_min = -5
    """v_max is the maximum velocity of the drone. In our units convention 0.25 is 25m/s"""
    v_max = 0.25
    pso(n, w, c1, c2, it, x_max, x_min, y_max, y_min, v_max)



    # probw=[]
    # for j in range(len(c2)):
    #     p=0
    #     for i in range(50):
    #         errors = pso(n,w,c1,c2[j],r1,r2,it,x_max,x_min,y_max,y_min,v_max)
    #         if errors[0] <= 2 or (errors[1]<=0.05 and errors[2] <= 0.05):
    #             p = p+1
    #         print("ERROR",errors)
    #     probw.append(round(p/50,2))
    #
    # print("probability with w",probw)
    # print(w)


















