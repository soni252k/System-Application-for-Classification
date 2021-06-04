#---------------------------------------------------------------------------------------------------------------

# Set the seed
import random
#random_state=10

def naiveBayesFunc(dataset, target):
    global Accuracy
    global Precision
    global Recall
    global F1Score
    global CM
    #---------------------------------------------------------------------------------------------------------------

    # import the basic libraries
    import pandas as pd
    import numpy as np
    import string
    import math

    # to scale the data
    from sklearn.preprocessing import StandardScaler

    # split the dataset
    from sklearn.model_selection import train_test_split

    # 1. Information Gain
    # 2. Gini Index
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    # 3. Naive Bayes
    from sklearn.naive_bayes import GaussianNB

    # 4. KNN
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    # 5. Random Forest
    from sklearn.ensemble import RandomForestClassifier

    # 6. Gradient Boost
    from sklearn.ensemble import GradientBoostingClassifier

    # for summary report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report


    import matplotlib.pyplot as plt

    # ignore warning message
    import warnings
    warnings.filterwarnings('ignore')

    #---------------------------------------------------------------------------------------------------------------

    """ IMPORT THE DATASET """

    # Get the link for Dataset
    DsLink = dataset

    # Import the dataset
    data = pd.read_csv(DsLink)

    #---------------------------------------------------------------------------------------------------------------

    """ PREPROCESSING THE DATA """

    data.dropna()

    #---------------------------------------------------------------------------------------------------------------

    """ SPLIT THE DATA """
    target= int(target) 
    X = data.iloc[:, :target]
    Y = pd.DataFrame(data.iloc[:, target])

    N = len(Y)

    # Split the data into train and test data
    #
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=10)

    # FEATURE SCALING
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)

    X_test = sc.fit_transform(X_test)
    X_test = pd.DataFrame(X_test)

    #---------------------------------------------------------------------------------------------------------------

    """CLASSIFIER 3 : NAIVE BAYES """

    # MODEL
    NBClassifier = GaussianNB()
    NBClassifier.fit(X_train,Y_train)
    #NBClassifier

    # TESTING THE MODEL

    Y_predNB = NBClassifier.predict(X_test) # store the prediction data

    # predNB = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_predNB})

    # ANALYSIS OF TEST RESULTS

    # count the number of mismatches
    # count = 0
    # for i in range(0, len(predNB)):
    #     if Y_predNB[i] != Y_test.values.ravel()[i]:
    #         count = count + 1
    # print("Count of Wrong Prediction : " ,count)

    # Confusion Matrix
    #plot_confusion_matrix(NBClassifier, X_test.todense(), Y_test, cmap=plt.cm.Blues)
    # Confusion Matrix
    CM= confusion_matrix(Y_test,Y_predNB)

    # Accuracy
    Accuracy=round(100 * accuracy_score(Y_test, Y_predNB), 3)

    #
    # Precision
    Precision= round(100 * precision_score(Y_test, Y_predNB, average='weighted'), 3)

    # Recall
    Recall= round(100 * recall_score(Y_test, Y_predNB, average='weighted'), 3)

    #F1 Score
    F1Score= round(100 * f1_score(Y_test, Y_predNB, average='weighted'), 3)
