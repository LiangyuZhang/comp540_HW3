import random
import numpy as np
import matplotlib.pyplot as plt
import utils
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import time
import music_utils
from sklearn import cross_validation
import sklearn.metrics


# TODO: Get the music dataset (CEFS representation) [use code from Hw2]
MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# select the CEPS or FFT representation

X,y = music_utils.read_ceps(genres,MUSIC_DIR)

# TODO: Split into train, validation and test sets 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_train, y_train, test_size=0.1)

# TODO: Use the validation set to tune hyperparameters for softmax classifier
# choose learning rate and regularization strength (use the code from softmax_hw.py)
batch_sizes = [200, 300, 400]
iterations = [1000,2000,3000]
learning_rates = [5e-7, 1e-6, 5e-6]
regularization_strengths = [1e2, 1e3,1e4, 1e5]
best_val = -1
best_softmax = None

for batch_idx, batch_size in enumerate(batch_sizes):
  for it_idx, iteration in enumerate(iterations):
    for learningrate in learning_rates:
      for regularization in regularization_strengths:
        softmax=SoftmaxClassifier()

        softmax.train(X_train,y_train,learningrate,reg=regularization, num_iters=iteration,batch_size=batch_size, verbose=True)
        y_pred_val=softmax.predict(X_val)
        current_val = np.mean(y_pred_val==y_val)
        if(current_val>best_val):
          best_val = current_val
          best_softmax = softmax
          best_learning_rate = learningrate
          best_reg = regularization
          best_iteration = iteration
          best_batch_size = batch_size

print "best batch size is ", best_batch_size
print "best iteration is ", best_iteration
print "best reg is ", best_reg
print "best learning rate is ", best_learning_rate



# TODO: Evaluate best softmax classifier on set aside test set (use the code from softmax_hw.py)

if best_softmax:
  y_test_pred = best_softmax.predict(X_test)
  test_accuracy = np.mean(y_test == y_test_pred)
  print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

  print sklearn.metrics.confusion_matrix(y_test,y_test_pred)
  print sklearn.metrics.classification_report(y_test,y_test_pred)

# TODO: Compare performance against OVA classifier of Homework 2 with the same
# train, validation and test sets (use sklearn's classifier evaluation metrics)
