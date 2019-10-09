# Neural-Robot-Arm
A proof of concept to solve a robots inverse kinematics using neural networks.

<p align="middle">
  <img src="/images/Predicted TCP results.png" alt="Predicted TCP results" width=300>
</p>


## Introduction
It is common for roboticist to manually solve inverse kinematic equations. This usually requires high amounts of effort and experience, and usually in the end you have still have to program and test those equations.

I propose using ANN for obtaining the angles using the easily obtainable forward kinematic equations as the sample data generators.

<p align="middle">
  <img src="/images/Robots workspace.png" alt="Robots workspace" width=300>
</p>


## How to use
Open a terminal and run the python script as follows:

    python Neural_model_trainer

By default the following Neural Network is implemented:

<p align="middle">
  <img src="/images/ANN Model.png" alt="Default ANN Model" width=300>
</p>


## Future functionalities
In future versions I would like to add more robot models. So far, my plan is to add the following:

* Three joint robot
* Six joint robot
* XY axis plotter


## Requirements
You should install the following:

* Python
* Matplotlib
* Numpy
* Pandas
* Keras








