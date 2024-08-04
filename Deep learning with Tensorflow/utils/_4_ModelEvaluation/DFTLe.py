import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.DFTLi import *
from utils._2_Cleaning.DFTLc import *
from utils._3_ModelTraining.DFTLt import *

def check_best_learningrate(x_train,y_train):
    """
     Check the best learning rate for the model. This is a test function to make sure that we have a good learning rate
     
     @param x_train - training data of the model
     @param y_train - training labels of the model ( same as x
    """
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # add an extra layer
    tf.keras.layers.Dense(1) # output layer
    ])

    # Compile the model
    model_1.compile(loss="binary_crossentropy", # we can use strings here too
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["accuracy"])

    # Create a learning rate scheduler callback
    # traverse a set of learning rate values starting from 1e-3, increasing by 10**(epoch/20) every epoch
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch/3)
    )

    # Fit the model (passing the lr_scheduler callback)
    history = model_1.fit(x_train,
                        y_train,
                        epochs=100,
                        verbose=0,
                        callbacks=[lr_scheduler])
    
    # Plot the learning rate versus the loss
    lrs = 1e-5 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss");
    plt.show()
    print("CHART PLOT SUCCESSFUL")

def predict_testset_using_activatedfunctions(model_1,x_test,y_test):
    """
     Predicts testset using activation functions and prints accuracy. This is a wrapper for tf. predict and tf. evaluate
     
     @param model_1 - model to be used for prediction
     @param x_test - x values of the test set to be predicted
     @param y_test - y values of the test set to be
    """
    print("Predictions using Sigmoid ")
    y_preds = tf.round(model_1.predict(x_test))
    print(y_preds[:5])
    print(accuracy_score(y_test, y_preds))
    #model_1.evaluate(y_test, y_preds)
def loss_curves_plot(history):
    """
     Plot loss curves for each model. This is a convenience function to make it easier to visualize the loss curves in a way that is more readable.
     
     @param history - A History object that contains the history of the model
    """
    pd.DataFrame(history.history).plot()
    plt.title("Model_1 training curves")
    plt.show()
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/employee_attrition.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    X,Y,X_scaled,x_train,x_test,y_train,y_test = prepare_and_splitdata(df)
    model_1 = train_Tensorflowmodel_run1(x_train,y_train)
    model_1 = train_Tensorflowmodel_moreEpochs(model_1,x_train,y_train)
    model_1 = train_Tensorflowmodel_MoreLayers(x_train, y_train)
    model_1 = train_Tensorflowmodel_MoreNeurons_and_Morelayers(x_train,y_train)
    model_1,history = activationfunctions_with_model(x_train,y_train)
    check_best_learningrate(x_train,y_train)
    predict_testset_using_activatedfunctions(model_1,x_test,y_test)
    loss_curves_plot(history)
    
    