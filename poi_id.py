# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:01:49 2016

@author: Gaoyuan
"""

#!/usr/bin/python

import pickle
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from tester import dump_classifier_and_data, main
from sklearn.metrics.scorer import f1_scorer
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")


# ## Task 1: Select what features you'll use.
# ## features_list is a list of strings, each of which is a feature name.
# ## The first feature must be "poi".
features_list = ['poi', 'expenses', 'shared_receipt_with_poi', 'other','exercised_stock_options']


# ## Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# ## Task 2: Remove outliers

'''
# 查找outlier #
data_dict_df = pd.DataFrame(data_dict).T
data_dict_df = data_dict_df.drop('email_address', axis = 1)
data_dict_df = pd.DataFrame(data_dict_df, dtype = float)
data_dict_df.info()
data_dict_df.plot(x = 'salary', y = 'bonus', kind = 'scatter')
print data_dict_df['salary'].argmax()
'''

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
    
# ## Task 3: Create new feature(s)
def computeFraction(poi_messages, all_messages ):
    fraction = 0.
    if poi_messages == 0 or poi_messages == 'NaN' or all_messages == 0 or all_messages == 'NaN':
        fraction = 0
    else:
        fraction = float(poi_messages)/float(all_messages)
    return fraction

for e in data_dict.values():
    e['fraction_from_poi'] = computeFraction(e['from_poi_to_this_person'], e['to_messages'])
    e['fraction_to_poi'] = computeFraction(e['from_this_person_to_poi'], e['from_messages'])
    
# ## Store to my_dataset for easy export below.
my_dataset = data_dict

# ## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


def get_feature_importances_and_model_performance(features_list, clf):
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    folds = 1000
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    feature_importances = np.zeros(len(features_list)-1)
    precision = []
    recall = []
    f1_score = []
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        precision.append (metrics.precision_score(labels_test, pred))
        recall.append( metrics.recall_score(labels_test, pred))
        f1_score.append( metrics.f1_score(labels_test, pred))
        feature_importances = feature_importances + np.array(clf.feature_importances_)
    mean_feature_importances = feature_importances / folds
    print 'Mean feature importances:'
    print sorted(zip(map(lambda x: round(x, 4), mean_feature_importances), features_list[1:]), reverse=True)
    print 'precision: %f, recall: %f, f1_score: %f' %(np.mean(precision), np.mean(recall), np.mean(f1_score))

'''
clf = DecisionTreeClassifier()
features_list_0 = ['poi', 'expenses', 'shared_receipt_with_poi', 'other','exercised_stock_options']
get_feature_importances_and_model_performance(features_list_0, clf)
'''



#print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), features_list[1:]), reverse=True) #print dt.feature_importances_
#print clf.feature_importances_
# ## Task 4: Try a varity of classifiers
# ## Please name your classifier clf for easy export below.
# ## Note that if you want to do PCA or other multi-stage operations,
# ## you'll need to use Pipelines. For more info:
# ## http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
'''
clf = DecisionTreeClassifier()
dump_classifier_and_data(clf, my_dataset, features_list)
main()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
dump_classifier_and_data(clf, my_dataset, features_list)
main()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, features_list)
main()
'''
# ## Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ## using our testing script. Check the tester.py script in the final project
# ## folder for details on the evaluation method, especially the test_classifier
# ## function. Because of the small size of the dataset, the script uses
# ## stratified shuffle split cross validation. For more info: 
# ## http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#   features_train, features_test, labels_train, labels_test = \
#   train_test_split(features, labels, test_size=0.3, random_state=42)



#首先调整参数criterion：
'''
criterions = ['gini', 'entropy']
for e in criterions:
    clf = DecisionTreeClassifier(criterion = e)
    dump_classifier_and_data(clf, my_dataset, features_list)
    main()
'''
# 参数criterion 中 ‘gini’表现更好, 调整splitter：
'''
splitters = ['best', 'random']
for e in splitters:
    clf = DecisionTreeClassifier(criterion='gini', splitter = e)
    dump_classifier_and_data(clf, my_dataset, features_list)
    main()
'''

# 参数splitter 中 ‘best’表现更好, 调整 max_features：
'''
max_features = [2, 3, 4, None]
for e in max_features:
    clf = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_features = e)
    dump_classifier_and_data(clf, my_dataset, features_list)
    main()
'''
# 参数max_features的不同取值差别不明显，因此保留默认值 None。调整 max_depth：
'''
max_depth = [2,3,4,5,6,7,8,9,10, None]
for e in max_depth:
    clf = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth = e)
    dump_classifier_and_data(clf, my_dataset, features_list)
    main()
'''

# 参数max_depth = 8 时表现最好。调整 min_samples_split：
'''
min_samples_split = [2,3,4,5]
for e in min_samples_split:
    clf = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth = 8, min_samples_split = e)
    dump_classifier_and_data(clf, my_dataset, features_list)    
    main()
'''
# 参数min_samples_split的不同取值差别不明显，因此保留默认值 2。
#最终经过参数调整的分类器为：

clf = DecisionTreeClassifier(criterion='gini', splitter = 'best', max_depth = 8)

#对模型进行交叉验证，使用StratifiedShuffleSplit方法划分训练集和测试集
#相关代码已包含在get_feature_importances_and_model_performance函数中
get_feature_importances_and_model_performance(features_list, clf)

# ## Task 6: Dump your classifier, dataset, and features_list so anyone can
# ## check your results. You do not need to change anything below, but make sure
# ## that the version of poi_id.py that you submit can be run on its own and
# ## generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

