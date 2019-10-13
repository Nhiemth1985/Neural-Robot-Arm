
##############################
#                            #
# Created by: Daniel Aguirre #
# Date: 13/10/2019           #
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
import Robot_n_joints
from sklearn.model_selection import train_test_split

# Load data
def load_data(n_links):
    
    df_train = pd.read_pickle("sample_data/sample_points_2arm.pkl")
    x = df_train.iloc[:,-2:] # Thetas
    y = df_train.iloc[:,0:-2] # X and Y    
    x_train, x_vali, y_train, y_vali = train_test_split(x, y, test_size=0.2, random_state=1)

    return x_train, y_train, x_vali, y_vali
    

# Create the NN model
def create_model(output_dim):

    model = keras.Sequential()
    model.add(keras.layers.Dense(100, use_bias=True, activation="relu",input_dim=2))
    model.add(keras.layers.Dense(100, use_bias=True, activation="relu"))
    model.add(keras.layers.Dense(100, use_bias=True, activation="relu"))
    model.add(keras.layers.Dense(output_dim, use_bias=True, activation="linear"))

    model.compile(optimizer="adam",
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_vali, y_vali):

    print(x_train)
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
    loss_and_metrics = model.evaluate(x_vali, y_vali, batch_size=128)

    print("----------------------------")
    print("\n      Loss_and_metrics      ")
    print(loss_and_metrics)
    print("----------------------------\n")

    y_predic = model.predict(x_vali, batch_size=128)
    diff = y_predic - y_vali

    #print("----------------------------")
    #print(" Max Diff (degree) ")
    #print(diff.max()*180/math.pi)
    #print(diff.min()*180/math.pi)
    #print("----------------------------")


def test_model_circle(model, robot):
    
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
    print(y_predic)
    for i in range(len(y_predic)):

        thetas = []
        for j in range(robot.n_links):
            thetas.append(y_predic[i,j]) 
        
        xpos, ypos = robot.forwardKinematics(thetas)
        data_predic[i,:] = [xpos,ypos]
        
    # Plot the results
    xpos_target = data_circle[:,0]
    ypos_target = data_circle[:,1]

    xpos_predic = data_predic[:,0]
    ypos_predic = data_predic[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xpos_target, ypos_target)
    ax.scatter(xpos_predic, ypos_predic)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Robots TCP")
    plt.legend(("Target","Predicted"))
    ax.grid(True)
    plt.show(block = False)

    # Plot the difference
    diff_xpos = xpos_target - xpos_predic
    diff_ypos = ypos_target - ypos_predic

    num_bins = 5

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.hist(diff_xpos, num_bins, facecolor='blue')
    ax1.legend(["Diff in x"],loc="upper center")
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.hist(diff_ypos, num_bins, facecolor='orange')
    ax2.legend(["Diff in y"],loc="upper center")

    plt.xlabel("Difference (Target - Predicted) (m)")
    plt.show()


def getSampleData(robot, sample):    

    # Generate sample points    
    data = np.zeros((samples,(robot.n_links + 2)))

    for i in range(samples):
        thetas = []
        for j in range(robot.n_links):
            thetas.append(random.random()*math.pi/2)

        x, y = robot.forwardKinematics(thetas)
        data[i,:] = (thetas + [x] + [y])

    # Save the results into a csv file
    column_names = []
    for i in range(robot.n_links):
        column_names.append("theta" + str(i+1))

    
    column_names = column_names + ["x"] + ["y"]
    print(column_names)
    df = pd.DataFrame(data,columns=column_names)
    df.to_pickle("sample_data/sample_points_2arm.pkl")
    
    return data


def plotSampleData(data):
    # Plot the results
    x = data[:,-2]
    y = data[:,-1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Robots workspace")
    ax.grid(True)
    plt.show(block = False)

if __name__ == "__main__":

    # Define robot model to use    
    robot = Robot_n_joints.robot_n_joints(2)

    # Generate data
    samples = 8000
    data = getSampleData(robot, samples)
    plotSampleData(data)
    
    # Load data
    x_train, y_train, x_vali, y_vali = load_data(robot.n_links)

    # Create the NN model
    model = create_model(robot.n_links)
    model.summary()

    # Train the model
    train_model(model, x_train, y_train, x_vali, y_vali)

    # Validate the model with a profile
    test_model_circle(model, robot)

    

    







