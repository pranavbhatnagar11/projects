"""
Pranav Bhatnagar
Class: CS 677 - Spring 2 - 2021
Date: 2021/08/15
Project 
"""

#importing relevant libraries
import pandas as pd
import sys
import warnings
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
        
filename ='water_potability.csv'

#creating data frame from a csv file
df = pd.read_csv(filename)
pd.set_option('display.max_columns', 11)

#test if there are any missing values
# for i in df.columns:
#     print(i,end=': ')
#     print(sum(df[i].isna()))
   

#creating histogram for all independent variables
df.hist(bins=30, figsize=(8,8))

#creating correlation matrix
plt.figure()
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
    
#check for class imbalance
print(df['Potability'].value_counts())

def metrics(df):
    '''This function takes data frame as an argument, and returns
    accuracy, fscore, precision, recall, true positives, false 
    positives, false negatives, true negatives, true positive rate,
    true negative rate'''
    
    tp = df[(df['y_test']==1) & (df['y_pred']==1)].shape[0]

    fp = df[(df['y_test']==0) & (df['y_pred']==1)].shape[0]

    fn = df[(df['y_test']==1) & (df['y_pred']==0)].shape[0]

    tn = df[(df['y_test']==0) & (df['y_pred']==0)].shape[0]
    
    acc = (tp+tn)/(tp+tn+fn+fp)

    recall = tp/(tp+fn)
    
    precision = tp/(tp+fp)
    
    fscore = (2*precision*recall)/(precision+recall)
    
    tpr = tp/(tp+fn)

    tnr = tn/(tn+fp)
    
    return(acc, fscore, precision, recall, tp, fp, fn, tn, tpr, tnr)


#group by the class for each variable 
# for i in df.columns[:-1]:
#     print(df.groupby("Potability")[i].mean())


#filling missing values by mean of class label
df['ph'] = df.groupby(['Potability'])['ph']\
    .transform(lambda x: x.fillna(x.mean()))
df['Sulfate'] = df.groupby(['Potability'])['Sulfate']\
    .transform(lambda x: x.fillna(x.mean()))
df['Trihalomethanes'] = df.groupby(['Potability'])['Trihalomethanes']\
    .transform(lambda x: x.fillna(x.mean()))

#test if there are any missing values
# for i in df.columns:
#     print(i,end=': ')
#     print(sum(df[i].isna()))

#selecting X and y
X = df.iloc[:,0:-1]
y = df['Potability']

#splitting into train and test
X_train,X_test,y_train,y_test = \
    train_test_split(X, y, test_size=0.3, random_state=4)


##Classifiers

#KNN
scaler = StandardScaler()
scaler.fit(X_train)

#standardizing on training and testing set
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

all_metrics_list = []

#k values 
k_vals = range(3,21,2)

for k in k_vals:
    #run knn for different neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    
    #predict using knn
    pred_k = knn_classifier.predict(X_test_scaled)
    
    #creating data frame with true and predicted labels
    df_knn_y = pd.DataFrame({'y_test':y_test, 'y_pred':pred_k})
    
    #appending all accuracy scores
    all_metrics_list.append(metrics(df_knn_y))

#unzipping all metrics
unzipped_all_metrics = zip(*all_metrics_list)
all_metrics = list(unzipped_all_metrics)

print('-'*50)    
print('KNN')
#creating data frame with all metrics 
df_knn = pd.DataFrame({'k':k_vals, 'Accuracy':all_metrics[0], 
                       'FScore':all_metrics[1],
                       'Precision':all_metrics[2], 
                       'Recall':all_metrics[3],
                       'True Positive':all_metrics[4],
                       'False Positive':all_metrics[5],
                       'False Negative':all_metrics[6],
                       'True Negative':all_metrics[7],
                       'True Positive Rate':all_metrics[8],
                       'True Negative Rate':all_metrics[9]})
print(df_knn)



#Logistic Regression

print('-'*50)
print('Logistic Regression')
penalty = ['l1','l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

#hyperparameters
grid = dict(penalty=penalty,C=c_values)

#applying gridsearch to find the best parameters in training
clf = GridSearchCV(estimator=LogisticRegression(), cv = 8, \
                   param_grid=grid, n_jobs=-1, scoring='precision')
grid_result = clf.fit(X_train_scaled,y_train)

#best results
print("Best Score: %f using %s" % (grid_result.best_score_, \
                                   grid_result.best_params_))
print(grid_result.best_estimator_)

#predicting using logistic regression
pred_log = grid_result.best_estimator_.predict(X_test_scaled)

#creating data frame with true and predicted labels
df_log_y = pd.DataFrame({'y_test':y_test, 'y_pred':pred_log})

#list of all metrics
all_metrics = list(metrics(df_log_y))
 
#creating data frame with all metrics 
data_log = {'Accuracy':all_metrics[0], 
            'FScore':all_metrics[1],
            'Precision':all_metrics[2], 
            'Recall':all_metrics[3],
            'True Positive':all_metrics[4],
            'False Positive':all_metrics[5],
            'False Negative':all_metrics[6],
            'True Negative':all_metrics[7],
            'True Positive Rate':all_metrics[8],
            'True Negative Rate':all_metrics[9]}

df_log = pd.Series(data_log).to_frame('Logistic Regression Metrics')
print(df_log)



print('-'*50)    
print('Naive Bayesian')

NB_classifier = GaussianNB().fit(X_train,y_train)

#predicting using naive bayes
y_pred = NB_classifier.predict(X_test)

#creating data frame with true and predicted labels
df_naive_y = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

all_metrics = list(metrics(df_naive_y))

#creating data frame with all metrics 
data_nb = {'Accuracy':all_metrics[0], 
            'FScore':all_metrics[1],
            'Precision':all_metrics[2], 
            'Recall':all_metrics[3],
            'True Positive':all_metrics[4],
            'False Positive':all_metrics[5],
            'False Negative':all_metrics[6],
            'True Negative':all_metrics[7],
            'True Positive Rate':all_metrics[8],
            'True Negative Rate':all_metrics[9]}

df_nb = pd.Series(data_nb).to_frame('Naive Bayes Metrics')
print(df_nb)
    
    
    
print('-'*50) 
print('Decision Tree')

#hyperparameters
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]
   
#applying gridsearch to find the best parameters in training 
grid = dict(criterion=criterion,max_depth=max_depth)
clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(), cv = 8, \
                   param_grid=grid, n_jobs=-1, scoring='precision',\
                   error_score=0)
grid_result = clf.fit(X_train_scaled,y_train)

#best results
print("Best Score: %f using %s" % (grid_result.best_score_, \
                                   grid_result.best_params_))
print(grid_result.best_estimator_)


tree_classifier = grid_result.best_estimator_.fit(X_train_scaled,y_train)

#predicting using decision tree
y_pred = tree_classifier.predict(X_test_scaled)

#creating data frame with true and predicted labels
df_tree_y = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

all_metrics = list(metrics(df_tree_y))

#creating data frame with all metrics 
data_dt = {'Accuracy':all_metrics[0], 
            'FScore':all_metrics[1],
            'Precision':all_metrics[2], 
            'Recall':all_metrics[3],
            'True Positive':all_metrics[4],
            'False Positive':all_metrics[5],
            'False Negative':all_metrics[6],
            'True Negative':all_metrics[7],
            'True Positive Rate':all_metrics[8],
            'True Negative Rate':all_metrics[9]}

df_dt = pd.Series(data_dt).to_frame('Decision Tree Metrics')
print(df_dt)



print('-'*50)     
print('Random Forest')

#hyperparameters
n_estimators = [10,200,500,1000]
max_depth = [2,6,10]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

#applying gridsearch to find the best parameters in training
# grid = dict(n_estimators=n_estimators,max_depth=max_depth, \
#             min_samples_split=min_samples_split,\
#             min_samples_leaf=min_samples_leaf)
# clf = GridSearchCV(estimator=RandomForestClassifier(), cv = 8, \
#                     param_grid=grid, n_jobs=-1, scoring='precision',\
#                     error_score=0)
# grid_result = clf.fit(X_train_scaled,y_train)

# best results
# print("Best Score: %f using %s" % (grid_result.best_score_, \
#                                    grid_result.best_params_))
# print(grid_result.best_estimator_)

#random forest with best hyperparameters
rf_classifier = RandomForestClassifier(max_depth = 2, min_samples_leaf = 1, \
                                       min_samples_split = 5, \
                                       n_estimators = 500,\
                                       random_state = 11)

rf_classifier = rf_classifier.fit(X_train_scaled,y_train)

#predicting using random forest
y_pred = rf_classifier.predict(X_test_scaled)

#creating data frame with true and predicted labels
df_rf_y = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

all_metrics = list(metrics(df_rf_y))

#creating data frame with all metrics 
data_rf = {'Accuracy':all_metrics[0], 
            'FScore':all_metrics[1],
            'Precision':all_metrics[2], 
            'Recall':all_metrics[3],
            'True Positive':all_metrics[4],
            'False Positive':all_metrics[5],
            'False Negative':all_metrics[6],
            'True Negative':all_metrics[7],
            'True Positive Rate':all_metrics[8],
            'True Negative Rate':all_metrics[9]}

df_rf = pd.Series(data_rf).to_frame('Random Forest Metrics')
print(df_rf)
    
    

print('-'*50)
print('SVM')

#hyperparameters
C = [1, 10, 100]
gamma = [0.001, 0.0001]
kernel = ['linear','rbf']

#applying gridsearch to find the best parameters in training
# grid = dict(C=C,gamma=gamma,kernel=kernel)
# clf = GridSearchCV(estimator=svm.SVC(), cv = 8, param_grid=grid,\
#                    n_jobs=-1, scoring='precision',error_score=0)
# grid_result = clf.fit(X_train_scaled,y_train)

# best results
# print("Best Score: %f using %s" % (grid_result.best_score_, \
#                                    grid_result.best_params_))
# print(grid_result.best_estimator_)

#SVM with best hyperparameters
svm_classifier = svm.SVC(C=100,gamma = .001,kernel = 'rbf')
svm_classifier = svm_classifier.fit(X_train_scaled,y_train)

#predicting using SVM
y_pred = svm_classifier.predict(X_test_scaled)

#creating data frame with true and predicted labels
df_svc_y = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

all_metrics = list(metrics(df_svc_y))

#creating data frame with all metrics 
data_svm = {'Accuracy':all_metrics[0], 
            'FScore':all_metrics[1],
            'Precision':all_metrics[2], 
            'Recall':all_metrics[3],
            'True Positive':all_metrics[4],
            'False Positive':all_metrics[5],
            'False Negative':all_metrics[6],
            'True Negative':all_metrics[7],
            'True Positive Rate':all_metrics[8],
            'True Negative Rate':all_metrics[9]}

df_svm = pd.Series(data_svm).to_frame('SVM Metrics')
print(df_svm)
    




