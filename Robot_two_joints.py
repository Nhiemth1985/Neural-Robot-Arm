
##############################
#                            #
# Created by: Daniel Aguirre #
# Date: 06/10/2019           #
#                            #
##############################

import math
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd


# Two arm robot model
class robot_two_arms(object):
    def __init__(self, l1 = 0.5, l2 = 0.5):
        self.limits = ((0,math.pi/2),(0,math.pi/2))
        self.l1 = l1
        self.l2 = l2

    def forwardKinematics(self, theta1, theta2):

        # Check limits
        theta1 = self.checkLimit(theta1, self.limits[0])
        theta2 = self.checkLimit(theta2, self.limits[1])


        # Calculate trigonometrics values
        s1 = math.sin(theta1)
        c1 = math.cos(theta1)
        s12 = math.sin(theta1 + theta2)
        c12 = math.cos(theta1 + theta2)

        x = self.l1*c1 + self.l2*c12
        y = self.l1*s1 + self.l2*s12

        return x,y

    def inverseKinematicsAnalytical(self, x, y):
        pass

    def inverseKinematicsNN(self, x, y):
        pass

    def checkLimit(self, theta, limits):

        if theta > max(limits):
            theta = max(limits)
        elif theta < min(limits):
            theta = min(limits)

        return theta
        

if __name__ == "__main__":

    # Create robot model
    robot = robot_two_arms()

    # Generate and store sample points
    samples = 1000
    data = np.zeros((samples,4))

    random.seed(1337)
    for i in range(samples):
        theta1 = random.random()*math.pi/2
        theta2 = random.random()*math.pi/2

        x, y = robot.forwardKinematics(theta1, theta2)
        data[i,:] = (theta1, theta2, x, y)
        

    # Plot the results
    x = data[:,2]
    y = data[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.show()

    # Save the results into a csv file
    df = pd.DataFrame(data,columns=["theta1","theta2","x","y"])
    df.to_pickle("sample_data/sample_points_2arm.pkl")
