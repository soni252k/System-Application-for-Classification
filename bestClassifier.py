#---------------------------------------------------------------------------------------------------------------

# Set the seed
import random
#random.seed(a=None, version=2)

def bestClassifierFunc(dataset, target):
    global Accuracy
    global Precision

    Accuracy= [0,0,0,0,0,0,0]
    Precision= [0,0,0,0,0,0,0]

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
    # from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score

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

    """CLASSIFIER 1 : INFORMATION GAIN """

    # MODEL
    IGClassifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    IGClassifier.fit(X_train, Y_train)
    # IGClassifier

    # TESTING THE MODEL
    Y_predIG = IGClassifier.predict(X_test)

    # predIG = pd.DataFrame({'Actual' : Y_test.values.ravel(), 'Predicted' : Y_predIG})

    # ANALYSIS OF TEST RESULTS

    # Accuracy
    Accuracy[0]= round(100 * accuracy_score(Y_test, Y_predIG), 3)

    #
    # Precision
    Precision[0]= round(100 * precision_score(Y_test, Y_predIG, average='weighted'), 3)

    #---------------------------------------------------------------------------------------------------------------

    """CLASSIFIER 2 : GINI INDEX """

    GIClassifier=DecisionTreeClassifier(criterion='gini',max_depth=5)

    GIClassifier.fit(X_train,Y_train)
    #GIClassifier

    # TESTING THE MODEL w.r.t. BOW

    Y_predGI=GIClassifier.predict(X_test)

    # predGI = pd.DataFrame({'Actual' : Y_test.values.ravel(), 'Predicted' : Y_predGI})

    # ANALYSIS OF TEST RESULTS

    # Confusion Matrix
    #plot_confusion_matrix(GIClassifier, X_test, Y_test, cmap=plt.cm.Blues)

    # Accuracy
    Accuracy[1]= round(100 * accuracy_score(Y_test, Y_predGI), 3)
    
    # Precision
    Precision[1]= round(100 * precision_score(Y_test, Y_predGI, average='weighted'), 3)

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

    # Accuracy
    Accuracy[2]= round(100 * accuracy_score(Y_test, Y_predNB), 3)

    # Precision
    Precision[2]= round(100 * precision_score(Y_test, Y_predNB, average='weighted'), 3)

    # Recall
    #print("Recall : ", round(100 * recall_score(Y_test, Y_predNB, average='weighted'), 3))
    
    #F1 Score
    #print("F1 Score : ", round(100 * f1_score(Y_test, Y_predNB, average='weighted'), 3))

    #---------------------------------------------------------------------------------------------------------------

    """ CLASSIFIER 4 : KNN (K-NEAREST NEIGHBOUR) """

    def find_optimal_k(Xtrain,Ytrain, myList):

        #creating odd list of K for KNN
        #myList = list(range(0,40))
        neighbors = list(filter(lambda x: x % 2 != 0, myList))

        # empty list that will hold cv scores
        cv_scores = []

        # perform 10-fold cross validation
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, Xtrain, Ytrain, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())

        # changing to misclassification error
        MSE = [1 - x for x in cv_scores]

        # determining best k
        optimal_k = neighbors[MSE.index(min(MSE))]
        #print('\nThe optimal number of neighbors is %d.' % optimal_k)

        #
        # plt.figure(figsize=(10,6))
        # plt.plot(list(filter(lambda x: x % 2 != 0, myList)),MSE,color='blue', linestyle='dashed', marker='o',
        #          markerfacecolor='red', markersize=10)
        # plt.title('Error Rate vs. K Value')
        # plt.xlabel('K')
        # plt.ylabel('Error Rate')

        #print("the misclassification error for each k value is : ", np.round(MSE,3))

        return optimal_k

    n = int(math.sqrt(N))
    # TRAINING THE MODEL
    myList = list(range(0, n))
    optimal_k = find_optimal_k(X_train ,Y_train,myList)
    KNNClassifier = KNeighborsClassifier(n_neighbors = optimal_k)
    KNNClassifier.fit(X_train, Y_train)
    #KNNClassifier

    # TESTING THE MODEL w.r.t. BOW

    Y_predKNN = KNNClassifier.predict(X_test) # store the prediction data

    # predKNN = pd.DataFrame({'Actual' : Y_test, 'Predicted' : Y_predKNN})

    # ANALYSIS OF TEST RESULTS

    # count the number of mismatches
    # count = 0
    # for i in range(0, len(predKNN)):
    #     if Y_predKNN[i] != Y_test.values.ravel()[i]:
    #         count = count + 1
    #print("Count of Wrong Prediction : " ,count)

    # Confusion Matrix
    #plot_confusion_matrix(KNNClassifier, X_test.todense(), Y_test, cmap=plt.cm.Blues)

    # Accuracy
    Accuracy[3]=round(100 * accuracy_score(Y_test, Y_predKNN), 3)

    #
    # Precision
    Precision[3]= round(100 * precision_score(Y_test, Y_predKNN, average='weighted'), 3)

    # Recall
    #print("Recall : ", round(100 * recall_score(Y_test, Y_predKNN, average='weighted'), 3))

    #F1 Score
    #print("F1 Score : ", round(100 * f1_score(Y_test, Y_predKNN, average='weighted'), 3))

    # Summary
    #print(classification_report(Y_test, Y_predKNN))

    #---------------------------------------------------------------------------------------------------------------

    """ CLASSIFIER 5 : RANDOM FOREST """

    # Model Creation
    RFClassifier = RandomForestClassifier(n_estimators = 3, random_state = 0)
    RFClassifier.fit(X_train, Y_train)
    #RFClassifier

    # TESTING THE MODEL

    Y_predRF = RFClassifier.predict(X_test)

    # predRF = pd.DataFrame({'Actual' : Y_test.values.ravel(), 'Predicted' : Y_predRF})

    # ANALYSIS OF TEST RESULTS

    # count the number of mismatches
    # count = 0
    # for i in range(0, len(predRF)):
    #     if Y_predRF[i] != Y_test.values.ravel()[i]:
    #         count = count + 1
    #print("Count of Wrong Prediction : " ,count)

    # Confusion Matrix
    #plot_confusion_matrix(RFClassifier, X_test, Y_test, cmap=plt.cm.Blues)

    # Accuracy
    Accuracy[4]= round(100 * accuracy_score(Y_test, Y_predRF), 3)

    #
    # Precision
    Precision[4]= round(100 * precision_score(Y_test, Y_predRF, average='weighted'), 3)

    # Recall
    #print("Recall : ", round(100 * recall_score(Y_test, Y_predRF, average='weighted'), 3))

    #F1 Score
    #print("F1 Score : ", round(100 * f1_score(Y_test, Y_predRF, average='weighted'), 3))

    # Summary
    #print(classification_report(Y_test, Y_predRF))

    #---------------------------------------------------------------------------------------------------------------

    """ CLASSIFIER 6 : GRADIENT BOOST """

    # MODEL
    GBClassifier = GradientBoostingClassifier(max_depth=2,
        n_estimators=3,
        learning_rate=1.0
    )
    GBClassifier.fit(X_train, Y_train.values.ravel())
    #GBClassifier

    # TESTING THE MODEL

    Y_predGB = GBClassifier.predict(X_test)

    # predGB = pd.DataFrame({'Actual' : Y_test.values.ravel(), 'Predicted' : Y_predGB})

    # ANALYSIS OF TEST RESULTS

    # count the number of mismatches
    # count = 0
    # for i in range(0, len(predGB)):
    #     if Y_predGB[i] != Y_test.values.ravel()[i]:
    #         count = count + 1
    #print("Count of Wrong Prediction : " ,count)

    # Confusion Matrix
    # plot_confusion_matrix(GBClassifier, X_test, Y_test, cmap=plt.cm.Blues)

    # Accuracy
    Accuracy[5]= round(100 * accuracy_score(Y_test, Y_predGB), 3)

    # Precision
    Precision[5]= round(100 * precision_score(Y_test, Y_predGB, average='weighted'), 3)

    # Recall
    #print("Recall : ", round(100 * recall_score(Y_test, Y_predGB, average='weighted'), 3))
    #
    #F1 Score
    #print("F1 Score : ", round(100 * f1_score(Y_test, Y_predGB, average='weighted'), 3))

    # Summary
    #print(classification_report(Y_test, Y_predGB))

    #---------------------------------------------------------------------------------------------------------------

    max = Accuracy[0]

    global val
    val = 0

    # Loop through the array
    for i in range(0, len(Accuracy)):
        # Compare elements of array with max
       if(Accuracy[i] > max) :
         max = Accuracy[i]
         val = i

    print("\n")

    #---------------------------------------------------------------------------------------------------------------

    global Classifiers
    Classifiers = ["Information Gain", "Gini Index", "Naive Bayes", "K-Nearest Neighbour", "Random Forest", "Gradient Boost"]
