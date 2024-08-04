import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.uscI import *
from utils._2_Cleaning.uscC import *

def train_KmeansModel_with_2features(df):
    kmodel = KMeans(n_clusters=5).fit(df[['Annual_Income','Spending_Score']])
    kmodel.cluster_centers_
    df['Cluster'] = kmodel.labels_
    print("Kmeans with 2 features succefully trained")
    print(df['Cluster'].value_counts())
    return df

def train_KmeansModel_with_3features(df):
    k = range(3,9)
    K = []
    ss = []
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Age','Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
    Variables3 = pd.DataFrame({'cluster': K, 'Silhouette_Score':ss})
    print("Kmeans with 3 features succefully trained")
    print(Variables3.head(2))
    return Variables3

def elbowmethod_to_checkclusters(df):
    # try using a for loop
    k = range(3,9)
    K = []
    WCSS = []
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    K, WCSS
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    if wss is not None:
        print(" Successful: Store the number of clusters and their respective WSS scores in a dataframe")
    print(wss.head(2))
    return wss

def silhouttemethod_to_checkclusters(df,wss):
    k = range(3,9) # to loop from 3 to 8
    K = []         # to store the values of k
    ss = []        # to store respective silhouetter scores
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
    wss['Silhouette_Score']=ss
    print("Successful: Store the number of clusters and their respective silhouette scores in a dataframe")
    print(wss.head(2))
    return wss

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/mall_customers.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
    df = train_KmeansModel_with_2features(df)
    wss = elbowmethod_to_checkclusters(df)
    wss = silhouttemethod_to_checkclusters(df,wss)
    Variables3 = train_KmeansModel_with_3features(df)
