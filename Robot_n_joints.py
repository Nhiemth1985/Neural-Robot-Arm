
##############################
#                            #
# Created by: Daniel Aguirre #
# Date: 10/10/2019           #
#                            #
##############################

import math
import random
import numpy as np
import pandas as pd


# N joint robot model
class robot_n_joints(object):
    def __init__(self, n_links):
        
        self.n_links = n_links
        self.length = []
        for i in range(self.n_links):
            self.length.append(0.5)


        self.limits = []
        for i in range(self.n_links):
            self.limits.append([0,math.pi/2])

    def forwardKinematics(self, thetas):

        # Check thetas dimensions matches
        if len(thetas) != self.n_links:
            thetas_temp = []
            for i in range(self.n_links):
                thetas_temp.append(thetas[i])

            thetas = thetas_temp

        # Check limits
        for i in range(self.n_links):
            thetas[i] = self.checkLimit(thetas[i], self.limits[i])
        

        # Calculate transformation matrixes
        ts = []
        for i in range(self.n_links):
            s = math.sin(thetas[i])
            c = math.cos(thetas[i])

            t = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            ts.append(t) 


        # Calculate the relative position vectors
        joint_poss = []
        for i in range(self.n_links):
            pos = np.array([[self.length[i]],[0],[0]])
            for j in range(i + 1):
                pos = ts[i - j].dot(pos)

            joint_poss.append(pos)

        # Calculate the TCP
        tcp = np.zeros([3,1])
        for i in range(self.n_links):
            tcp = tcp + joint_poss[i]

        x = tcp[0]
        y = tcp[1]
        z = tcp[2]

        return x,y

    def inverseKinematicsAnalytical(self, x, y):
        pass

    def inverseKinematicsNN(self, x, y):
        pass

    def loadNNModel(self, architecture, weights):
        pass

    def checkLimit(self, theta, limits):

        theta = self.checkAngle(theta)

        if theta > max(limits):
            theta = max(limits)
        elif theta < min(limits):
            theta = min(limits)

        return theta

    def checkAngle(self, theta):

        theta = theta % (2*math.pi)

        return theta
        




