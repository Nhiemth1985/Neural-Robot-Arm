
##############################
#                            #
# Created by: Daniel Aguirre #
# Date: 06/10/2019           #
#                            #
##############################

import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import keras

# Load data
df_train = pd.read_pickle("sample_data/sample_points_2arm.pkl")
x_train = df_train[["theta1","theta2"]]
y_train = df_train[["x","y"]]

df_test = pd.read_pickle("sample_data/sample_points_2arm_test.pkl")
x_test = df_test[["theta1","theta2"]]
y_test = df_test[["x","y"]]

# Create the NN model
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(3,use_bias=True, activation="linear",input_dim=2))
    model.add(keras.layers.Dense(100,use_bias=True, activation="tanh"))
    model.add(keras.layers.Dense(2,use_bias=True, activation="linear"))
    
    model.compile(optimizer="adam",
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

simple_model = create_model()
simple_model.summary()



# Train the model
simple_model.fit(x_train, y_train, epochs=5, batch_size=32)



# Validation
loss_and_metrics = simple_model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
#simple_model.predict(x_test, batch_size=128)






