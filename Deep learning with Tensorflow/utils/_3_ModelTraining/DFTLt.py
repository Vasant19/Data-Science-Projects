import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.DFTLi import *
from utils._2_Cleaning.DFTLc import *

def train_Tensorflowmodel_run1(x_train, y_train):
    """
     Train a Tensorflow model using Sequential API and evaluate the model on the training data
     
     @param x_train - The training data to train the model on
     @param y_train - The training labels to train the model on
     
     @return The model that was trained on the training data and
    """
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # 1. Create the model using the Sequential API
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1) #output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), # binary since we are working with 2 clases (0 & 1)
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=5,verbose=0)
    print("Tensorflow First evaluation using Sequential API:")
    model_1.evaluate(x_train, y_train)
    return model_1

def train_Tensorflowmodel_moreEpochs(model_1,x_train,y_train):
    """
     Train a Tensorflow model for more epochs and evaluate. This is useful for debugging
     
     @param model_1 - The model to train.
     @param x_train - The training data for the model.
     @param y_train - The training labels for the model.
     
     @return The trained model with evaluation done in batches of 100
    """
    tf.keras.utils.set_random_seed(42)
    print("Tensorflow Second evaluation using more Epochs:")
    # Train our model for longer (more chances to look at the data)
    history = model_1.fit(x_train, y_train, epochs=100, verbose=0) # set verbose=0 to remove training updates
    model_1.evaluate(x_train, y_train)
    return model_1

def train_Tensorflowmodel_MoreLayers(x_train, y_train):
    """
     Trains a Tensorflow model with more layers. In this case we are going to use tf. keras. Sequential to compile and evaluate the model
     
     @param x_train - training data of shape [ batch_size d1... dN ]
     @param y_train - target data of shape [ batch_size d1... dN ]
     
     @return model_1 trained and eval'd Keras
    """
    tf.keras.utils.set_random_seed(42)
    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # add an extra layer
    tf.keras.layers.Dense(1) # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    print("Tensorflow Third evaluation using more Layers:")
    model_1.evaluate(x_train, y_train)
    return model_1

def train_Tensorflowmodel_MoreNeurons_and_Morelayers(x_train,y_train):
    """
     Train a Tensorflow model with more Layers and Neurons and evaluate the model
     
     @param x_train - training data for the model
     @param y_train - target data for the model ( labels
    """
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1), # add another layer with 1 neuron
    tf.keras.layers.Dense(1) # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    print("Tensorflow Fourth evaluation using more Layers and Neurons:")
    model_1.evaluate(x_train, y_train)
    return model_1

def activationfunctions_with_model(x_train,y_train):
    """
     This function uses a Keras model to train the neural network. In this case we are going to create a Sequential model with two layers that are different from the one used for training and one that is compiled and the output layer is a sigmoid layer
     
     @param x_train - input data of the network
     @param y_train - target data of the network ( y_train [ i ] = x_train [ i ]
     
     @return history of the model
    """
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # try activations LeakyReLU, sigmoid, Relu, tanh. Default is Linear
    tf.keras.layers.Dense(1, activation = 'sigmoid') # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                    metrics=['accuracy'])

    # 3. Fit the model
    history = model_1.fit(x_train, y_train, epochs=50,verbose=0)
    print("Tensorflow Fifth evaluation using Activation Function")
    model_1.evaluate(x_train, y_train)
    return model_1,history


if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/employee_attrition.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    X,Y,X_scaled,x_train,x_test,y_train,y_test = prepare_and_splitdata(df)
    model_1 = train_Tensorflowmodel_run1(x_train,y_train)
    train_Tensorflowmodel_moreEpochs(model_1,x_train,y_train)
    train_Tensorflowmodel_MoreLayers(x_train, y_train)
    train_Tensorflowmodel_MoreNeurons_and_Morelayers(x_train,y_train)
    activationfunctions_with_model(x_train,y_train)
    
