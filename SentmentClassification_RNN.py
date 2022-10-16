
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import math
import os
import random
import tarfile
import re

from six.moves import urllib
import six
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import tensorflow as tf

print(np.__version__)
print(mp.__version__)
print(tf.__version__)


# In[2]:


DOWNLOADED_FILE = 'ImdbReviews.tar.gz'
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")


# In[3]:


def get_reviews(dirname,positive=True):
    label = 1 if positive else 0
    reviews = []
    labels = []
    
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            with open (dirname+filename, 'r') as f:
                review = f.read()
                review = review.lower().replace("<br />"," ")
                review = re.sub(TOKEN_REGEX,'',review)
                reviews.append(review)
                labels.append(label)
    return reviews,labels


# In[4]:


def extract_labels_data():
    if not os.path.exists('aclimdb'):
        with tarfile.open(DOWNLOADED_FILE) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()
        
    positive_reviews,positive_labels = get_reviews("/aclimdb/train/pos/",positive=True)
    negative_reviews,negative_labels = get_reviews("/aclimdb/train/neg/",positive=False)
    
    data = positive_reviews + negative_reviews
    labels = positive_labels+ negative_labels
    
    return data,labels


# In[5]:


data,labels = extract_labels_data()
# data,labels = data.decode('utf-8'),labels.decode('utf-8')


# In[16]:


data[3]


# In[17]:


x_data[3]


# In[7]:



MAX_SEQUENCE_LENGTH = 250

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)

x_data = np.array(list(vocab_processor.fit_transform(data)))
y_output = np.array(labels)

vocabulary_size = len(vocab_processor.vocabulary_)
print(vocabulary_size)


# In[268]:


x_data[3:5]


# In[8]:


np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]

####################################


TRAIN_DATA = 5000
TOTAL_DATA = 6000

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]
test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]



# In[9]:


tf.reset_default_graph()
x = tf.placeholder(tf.int32,[None,MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int32,[None])
num_epochs = 20
batch_size = 25
embedding_size = 50
max_label = 2 ##maximum label of target 0/1

################


embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-0.1,1.0))
embeddings = tf.nn.embedding_lookup(embedding_matrix,x)



# In[10]:


embedding_matrix ##size of vocab


# In[11]:


embeddings  ## embedding of current btch of data "?" = batch size,250 = no of instances in time 50 = dimensionality of single inut
# [None,n_steps,n_inputs]


# In[12]:


lstmcell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmcell,output_keep_prob=0.75)
_,(encoding,_) = tf.nn.dynamic_rnn(lstmcell,embeddings,dtype=tf.float32)


# In[13]:


encoding


# In[14]:


logits = tf.layers.dense(encoding,max_label,activation=None)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss = tf.reduce_mean(cross_entropy)

prediction = tf.equal(tf.argmax(logits,1),tf.cast(y,tf.int64))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)


# In[ ]:


num_epochs = 20

init = tf.global_variables_initializer()
with tf.Session() as session:
    init.run()

    for epoch in range(num_epochs):
        
        num_batches = int(len(train_data) // batch_size) + 1
        
        for i in range(num_batches):
            # Select train data
            min_ix = i * batch_size
            max_ix = np.min([len(train_data), ((i+1) * batch_size)])

            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]
            
            train_dict = {x: x_train_batch, y: y_train_batch}
            
            
            session.run(train_step, feed_dict=train_dict)
            
            train_loss, train_acc = session.run([loss, accuracy], feed_dict=train_dict)

        test_dict = {x: test_data, y: test_target}
        test_loss, test_acc = session.run([loss, accuracy], feed_dict=test_dict)    
        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.5}'.format(epoch + 1, test_loss, test_acc)) 

