
##############################
#                            #
# Created by: Daniel Aguirre #
# Date: 06/10/2019           #
#                            #
##############################

import warnings
warnings.filterwarnings("ignore")

import math
import random
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import Robot_two_joints

# Load data
def load_data():
    df_train = pd.read_pickle("sample_data/sample_points_2arm.pkl")
    x_train = df_train[["x","y"]]
    y_train = df_train[["theta1","theta2"]]

    df_test = pd.read_pickle("sample_data/sample_points_2arm_test.pkl")
    x_test = df_test[["x","y"]]
    y_test = df_test[["theta1","theta2"]]

    return x_train, y_train, x_test, y_test

# Create the NN model
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(50, use_bias=True, activation="relu",input_dim=2))
    model.add(keras.layers.Dense(100, use_bias=True, activation="relu"))
    model.add(keras.layers.Dense(100, use_bias=True, activation="relu"))
    model.add(keras.layers.Dense(2, use_bias=True, activation="linear"))

    model.compile(optimizer="adam",
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test):

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Save the weights
    model.save_weights("weights/model.h5")
    print("Saved model to disk")

    # Save the architecture
    model_json = model.to_json()
    with open("architectures/model.json", "w") as json_file:
        json_file.write(model_json)


    # Validation: Random points
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

    print("----------------------------\n")
    print("\n      Loss_and_metrics      ")
    print(loss_and_metrics)
    print("----------------------------\n")

    y_predic = model.predict(x_test, batch_size=128)
    diff = y_predic - y_test

    #print("----- Max Diff (degree)-----")
    #print(diff.max()*180/math.pi)
    #print(diff.min()*180/math.pi)


def validate_model_circle(model, robot):
    
    # Generate the points
    point_num = 100
    data_circle = np.zeros((point_num,2))

    radious = 0.1
    x_dis = 0.5
    y_dis = 0.7
    angle = 0
    for i in range(point_num):    
        angle = 2*math.pi/point_num*i
        x = radious*math.cos(angle) + x_dis
        y = radious*math.sin(angle) + y_dis
        data_circle[i,:] = [x,y]

    # Predict the theta values        
    y_predic = model.predict(data_circle)#, batch_size=128)

    # Get the XY position with FK
    data_predic = np.zeros((len(y_predic),2))
    for i in range(len(y_predic)):
        theta1 = y_predic[i,0]
        theta2 = y_predic[i,1]
        xpos, ypos = robot.forwardKinematics(theta1, theta2)
        data_predic[i,:] = [xpos,ypos]
        
    # Plot the results
    xpos_target = data_circle[:,0]
    ypos_target = data_circle[:,1]

    xpos_predic = data_predic[:,0]
    ypos_predic = data_predic[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xpos_target, ypos_target)
    ax.scatter(xpos_predic, ypos_predic)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Robots TCP")
    plt.legend(("Target","Predicted"))
    ax.grid(True)
    plt.show()


def getSampleData(robot, sample):    

    # Generate sample points    
    data = np.zeros((samples,4))

    for i in range(samples):
        theta1 = random.random()*math.pi/2
        theta2 = random.random()*math.pi/2

        x, y = robot.forwardKinematics(theta1, theta2)
        data[i,:] = (theta1, theta2, x, y)

    # Save the results into a csv file
    df = pd.DataFrame(data,columns=["theta1","theta2","x","y"])
    df.to_pickle("sample_data/sample_points_2arm.pkl")

    return data


def plotSampleData(data):
    # Plot the results
    x = data[:,2]
    y = data[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Robots workspace")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":

    # Generate data
    robot = Robot_two_joints.robot_two_joint()
    samples = 2000

    data = getSampleData(robot, samples)
    plotSampleData(data)
    

    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Create the NN model
    model = create_model()
    model.summary()

    # Train the model
    train_model(model, x_train, y_train, x_test, y_test)

    
    # Validate the model with a profile
    validate_model_circle(model, robot)

    







