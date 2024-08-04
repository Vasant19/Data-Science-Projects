import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.uscI import *
from utils._2_Cleaning.uscC import *
from utils._3_ModelTraining.uscT import *

def evaluate_clusters_kmeanswith2features(df):
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.title("CLUSTER WITH 2 FEATURES")
    plt.show()

def evaluate_clusters_elbowmethod(wss):
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot cluster with 2 features')
    plt.show()
def evaluate_clusters_silhouettemethod(wss):  
    wss.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot for cluster with 2 features')
    plt.show()
    
def evaluate_clusters_kmeanswith3features(Variables3):
    Variables3.plot(x='cluster', y='Silhouette_Score')
    plt.title("Silhouette plot Cluster with 3 features")
    plt.show()
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/mall_customers.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
    df = train_KmeansModel_with_2features(df)
    evaluate_clusters_kmeanswith2features(df)
    wss = elbowmethod_to_checkclusters(df)
    evaluate_clusters_elbowmethod(wss)
    wss = silhouttemethod_to_checkclusters(df,wss) 
    evaluate_clusters_silhouettemethod(wss)
    Variables3 = train_KmeansModel_with_3features(df)
    evaluate_clusters_kmeanswith3features(Variables3)