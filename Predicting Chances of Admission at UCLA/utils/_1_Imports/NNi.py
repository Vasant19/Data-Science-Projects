import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")


# Import GridSearch CV
from sklearn.model_selection import GridSearchCV

