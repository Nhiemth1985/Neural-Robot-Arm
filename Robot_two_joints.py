
##############################
#                            #
# Created by: Daniel Aguirre #
# Date: 06/10/2019           #
#                            #
##############################

import math
import random
import numpy as np
import pandas as pd


# Two joint robot model
class robot_two_joint(object):
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
        

