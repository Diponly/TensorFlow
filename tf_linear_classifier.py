import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

census = pd.read_csv("census_data.csv")
census.head()
census.info()

'''

age               = int64
workclass         = non-null object
education         = non-null object
education_num     = non-null int64
marital_status    = non-null object
occupation        = non-null object
relationship      = non-null object
race              = non-null object
gender            = non-null object
capital_gain      = non-null int64
capital_loss      = non-null int64
hours_per_week    = non-null int64
native_country    = non-null object
income_bracket    = non-null object

We need to classiy the income_bracket which fall under <=50k or >50k

'''

def label_calc(label):
	if label == '<=50k':
		return 0
	else:
		return 1

census['income_bracket'] = census['income_bracket'].apply(label_calc)

'''
Perform a train split on the Data

'''

X_data = census.drop('income_bracket',axis=1)
y_labels = census['income_bracket']
X_train,X_test,y_train,y_test = train_test_split(X_data,y_labels,
	test_size=0.2,random_state=42)


'''
Use vocabulary lists or just use hash buckets.

hash_bucket_size = "this is being used 

                    when don't have any idea how many distinct value present !!"


'''

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",["Female","Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")


'''
collate everything as a feature col list
'''

feature_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)
model.train(input_fn=input_func,steps=6000)

'''
Evaluation of the test data

'''

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=100,shuffle=False)
predictions = list(model.predict(input_fn = pred_fn))
predictions[0]

final_preds = []for pred in predictions:
    final_preds.append(pred['class_ids'][0])

final_preds[:10]



from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))

























