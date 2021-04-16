import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report,f1_score,recall_score,precision_score,accuracy_score,precision_recall_curve,roc_curve,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
import pyrebase
from django.conf import settings
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse

firebase_storage = pyrebase.initialize_app(settings.CONFIG)
storage = firebase_storage.storage()
URL = storage.child("creditcard.csv").get_url(None)

LABELS = ["Normal", "Fraud"]

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import warnings
warnings.filterwarnings('ignore')

import random
random.seed(0)

def home(request):
    return JsonResponse({"API":"V1"},status=200)

def knn(request):
    cc_dataset = pd.read_csv(URL)
    cc_dataset.shape
    cc_dataset.head()
    cc_dataset.describe()
    cc_dataset.isnull().values.any()
    cc_dataset['Class'].value_counts()



    count_classes = pd.value_counts(cc_dataset['Class'], sort = True)
    count_classes.plot(kind = 'bar')
    plt.title("Transaction Class Distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.savefig('Fd_Nl.png')
    storage.child("Fd_Nl.png").put("Fd_Nl.png")
    Fd_Nl_URL = storage.child("Fd_Nl.png").get_url(None)
    #Splitting the input features and target label into different arrays
    X = cc_dataset.iloc[:,0:-1]
    Y = cc_dataset.iloc[:,-1]
    X.columns

    #Train Test split - By default train_test_split does STRATIFIED split based on label (y-value).
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    #Feature Scaling - Standardizing the scales for all x variables
    #PN: We should apply fit_transform() method on train set & only transform() method on test set
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)



    knn_clf = KNeighborsClassifier(n_neighbors=3)
    build_model_train_test(knn_clf,x_train,x_test,y_train,y_test)
    plt.savefig('knn_clf.png')
    storage.child("knn_clf.png").put("knn_clf.png")
    knn_clf_URL = storage.child("knn_clf.png").get_url(None)
    plt.close()

    return JsonResponse({"success":True,"Fd_Nl_URL":Fd_Nl_URL,"knn_clf_URL":knn_clf_URL},status=200)

def ada(request):

    cc_dataset = pd.read_csv(URL)
    cc_dataset.shape
    cc_dataset.head()
    cc_dataset.describe()
    cc_dataset.isnull().values.any()
    cc_dataset['Class'].value_counts()



    count_classes = pd.value_counts(cc_dataset['Class'], sort = True)
    count_classes.plot(kind = 'bar')
    plt.title("Transaction Class Distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.savefig('Fd_Nl.png')
    storage.child("Fd_Nl.png").put("Fd_Nl.png")
    Fd_Nl_URL = storage.child("Fd_Nl.png").get_url(None)

    #Splitting the input features and target label into different arrays
    X = cc_dataset.iloc[:,0:-1]
    Y = cc_dataset.iloc[:,-1]
    X.columns

    #Train Test split - By default train_test_split does STRATIFIED split based on label (y-value).
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    #Feature Scaling - Standardizing the scales for all x variables
    #PN: We should apply fit_transform() method on train set & only transform() method on test set
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)



    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3,class_weight='balanced'), n_estimators=100,
        algorithm="SAMME.R", learning_rate=0.5, random_state=0)
    build_model_train_test(ada_clf,x_train,x_test,y_train,y_test)
    plt.savefig('ada_clf.png')
    storage.child("ada_clf.png").put("ada_clf.png")
    ada_clf_URL = storage.child("ada_clf.png").get_url(None)
    plt.close()

    return JsonResponse({"success":True,"Fd_Nl_URL":Fd_Nl_URL,"ada_clf_URL":ada_clf_URL},status=200)

def rforest(request):

    cc_dataset = pd.read_csv(URL)
    cc_dataset.shape
    cc_dataset.head()
    cc_dataset.describe()
    cc_dataset.isnull().values.any()
    cc_dataset['Class'].value_counts()

    count_classes = pd.value_counts(cc_dataset['Class'], sort = True)
    count_classes.plot(kind = 'bar')
    plt.title("Transaction Class Distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.savefig('Fd_Nl.png')
    storage.child("Fd_Nl.png").put("Fd_Nl.png")
    Fd_Nl_URL = storage.child("Fd_Nl.png").get_url(None)

    #Splitting the input features and target label into different arrays
    X = cc_dataset.iloc[:,0:-1]
    Y = cc_dataset.iloc[:,-1]
    X.columns

    #Train Test split - By default train_test_split does STRATIFIED split based on label (y-value).
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    #Feature Scaling - Standardizing the scales for all x variables
    #PN: We should apply fit_transform() method on train set & only transform() method on test set
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    #Best estimator of random forest
    rnd_clf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=10, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=1,
                min_samples_split=5, min_weight_fraction_leaf=0.0,
                n_estimators=50, n_jobs=-1, oob_score=False, random_state=0,
                verbose=0, warm_start=False)
    build_model_train_test(rnd_clf,x_train,x_test,y_train,y_test)
    plt.savefig("rnd_clf.png")
    plt.close()
    storage.child("rnd_clf.png").put("rnd_clf.png")
    rnd_clf_URL = storage.child("rnd_clf.png").get_url(None)
    

    return JsonResponse({"success":True,"Fd_Nl_URL":Fd_Nl_URL,"rnd_clf_URL":rnd_clf_URL},status=200)

def conef(request):

    cc_dataset = pd.read_csv(URL)


    cc_dataset.shape

    cc_dataset.head()

    cc_dataset.describe()

    cc_dataset.isnull().values.any()

    cc_dataset['Class'].value_counts()



    count_classes = pd.value_counts(cc_dataset['Class'], sort = True)

    count_classes.plot(kind = 'bar')

    plt.title("Transaction Class Distribution")

    plt.xticks(range(2), LABELS)

    plt.xlabel("Class")

    plt.ylabel("Frequency")
    plt.savefig('Fd_Nl.png')
    storage.child("Fd_Nl.png").put("Fd_Nl.png")
    Fd_Nl_URL = storage.child("Fd_Nl.png").get_url(None)

    #Splitting the input features and target label into different arrays
    X = cc_dataset.iloc[:,0:-1]
    Y = cc_dataset.iloc[:,-1]
    X.columns

    #Train Test split - By default train_test_split does STRATIFIED split based on label (y-value).
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    #Feature Scaling - Standardizing the scales for all x variables
    #PN: We should apply fit_transform() method on train set & only transform() method on test set
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    #Best estimator of random forest
    rnd_clf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=10, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=1,
                min_samples_split=5, min_weight_fraction_leaf=0.0,
                n_estimators=50, n_jobs=-1, oob_score=False, random_state=0,
                verbose=0, warm_start=False)
    build_model_train_test(rnd_clf,x_train,x_test,y_train,y_test)
    plt.savefig("rnd_clf.png")
    storage.child("rnd_clf.png").put("rnd_clf.png")
    rnd_clf_URL = storage.child("rnd_clf.png").get_url(None)

    knn_clf = KNeighborsClassifier(n_neighbors=3)
    build_model_train_test(knn_clf,x_train,x_test,y_train,y_test)
    plt.savefig('knn_clf.png')
    storage.child("knn_clf.png").put("knn_clf.png")
    knn_clf_URL = storage.child("knn_clf.png").get_url(None)

    ada_clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=3,class_weight='balanced'), n_estimators=100,
            algorithm="SAMME.R", learning_rate=0.5, random_state=0)
    build_model_train_test(ada_clf,x_train,x_test,y_train,y_test)
    plt.savefig('ada_clf.png')
    storage.child("ada_clf.png").put("ada_clf.png")
    ada_clf_URL = storage.child("ada_clf.png").get_url(None)

    soft_voting_clf = VotingClassifier(
        estimators=[('rf', rnd_clf), ('ada', ada_clf), ('knn',knn_clf)], 
        voting='soft')

    build_model_train_test(soft_voting_clf,x_train,x_test,y_train,y_test)
    plt.savefig('soft_voting_clf.png')
    storage.child("soft_voting_clf.png").put("soft_voting_clf.png")
    soft_voting_clf_URL = storage.child("soft_voting_clf.png").get_url(None)

    probs_sv_test = soft_voting_clf.predict_proba(x_test)
    SelectThresholdByCV(probs_sv_test[:,1],y_test)


    y_pred_test = (probs_sv_test[:,1] > 0.571)
    Print_Accuracy_Scores(y_test,y_pred_test)

    plt.close()

    return JsonResponse({"success":True,"Fd_Nl_URL":Fd_Nl_URL,"knn_clf_URL":knn_clf_URL,"rnd_clf_URL":rnd_clf_URL,"soft_voting_clf_URL":soft_voting_clf_URL,"ada_clf_URL":ada_clf_URL},status=200)


def build_model_train_test(model,x_train,x_test,y_train,y_test):

    model.fit(x_train,y_train)

    y_pred = model.predict(x_train)

    print("\n----------Accuracy Scores on Train data------------------------------------")

    print("F1 Score: ", f1_score(y_train,y_pred))
    print("Precision Score: ", precision_score(y_train,y_pred))
    print("Recall Score: ", recall_score(y_train,y_pred))

    print("\n----------Accuracy Scores on Cross validation data------------------------------------")
    y_pred_cv = cross_val_predict(model,x_train,y_train,cv=5)
    print("F1 Score: ", f1_score(y_train,y_pred_cv))
    print("Precision Score: ", precision_score(y_train,y_pred_cv))
    print("Recall Score: ", recall_score(y_train,y_pred_cv))


    print("\n----------Accuracy Scores on Test data------------------------------------")
    y_pred_test = model.predict(x_test)

    print("F1 Score: ", f1_score(y_test,y_pred_test))
    print("Precision Score: ", precision_score(y_test,y_pred_test))
    print("Recall Score: ", recall_score(y_test,y_pred_test))

    #Confusion Matrix
    plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1,2)

    ax1 = plt.subplot(gs[0])
    cnf_matrix = confusion_matrix(y_train,y_pred,normalize='all')
    sns.heatmap(cnf_matrix,cmap='YlGnBu',annot=True)
    plt.title("Normalized Confusion Matrix - Train Data")

    ax3 = plt.subplot(gs[1])
    cnf_matrix = confusion_matrix(y_test,y_pred_test,normalize='all')
    sns.heatmap(cnf_matrix,cmap='YlGnBu',annot=True)
    plt.title("Normalized Confusion Matrix - Test Data")

    # plt.close()

def SelectThresholdByCV(probs,y):

    best_threshold = 0
    best_f1 = 0
    f = 0
    precision =0
    recall=0
    best_recall = 0
    best_precision = 0
    precisions=[]
    recalls=[]
    
    thresholds = np.arange(0.0,1.0,0.001)
    for threshold in thresholds:
        predictions = (probs > threshold)
        f = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        
        if f > best_f1:
            best_f1 = f
            best_precision = precision
            best_recall = recall
            best_threshold = threshold

        precisions.append(precision)
        recalls.append(recall)

    #Precision-Recall Trade-off
    plt.plot(thresholds,precisions,label='Precision')
    plt.plot(thresholds,recalls,label='Recall')
    plt.xlabel("Threshold")
    plt.title('Precision Recall Trade Off')
    plt.legend()
    # plt.show()
    plt.close()

    print ('Best F1 Score %f' %best_f1)
    print ('Best Precision Score %f' %best_precision)
    print ('Best Recall Score %f' %best_recall)
    print ('Best Epsilon Score', best_threshold)

def Print_Accuracy_Scores(y,y_pred):
    print("F1 Score: ", f1_score(y,y_pred))
    print("Precision Score: ", precision_score(y,y_pred))
    print("Recall Score: ", recall_score(y,y_pred))