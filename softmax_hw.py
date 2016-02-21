import random
import numpy as np
import matplotlib.pyplot as plt
import utils
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import time
from matplotlib import cm
import sklearn.metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Get the CIFAR-10 data broken up into train, validation and test sets

X_train, y_train, X_val, y_val, X_test, y_test = utils.get_CIFAR10_data()

# First implement the naive softmax loss function with nested loops.
# Open the file softmax.py and implement the
# softmax_loss_naive function.

# Generate a random softmax theta matrix and use it to compute the loss.

'''
theta = np.random.randn(3073,10) * 0.0001
loss, grad = softmax_loss_vectorized(theta, X_train, y_train, 0.0)

# Loss should be something close to - log(0.1)

print 'loss:', loss, ' should be close to ', - np.log(0.1)

# Use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient. (within 1e-7)

from gradient_check import grad_check_sparse
f = lambda th: softmax_loss_vectorized(th, X_train, y_train, 0.0)[0]
grad_numerical = grad_check_sparse(f, theta, grad, 10)

# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.

tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(theta, X_train, y_train, 0.00001)
toc = time.time()
print 'naive loss: %e computed in %fs' % (loss_naive, toc - tic)

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(theta, X_train, y_train, 0.00001)
toc = time.time()
print 'vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)


# We use the Frobenius norm to compare the two versions
# of the gradient.

grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print 'Loss difference: %f' % np.abs(loss_naive - loss_vectorized)
print 'Gradient difference: %f' % grad_difference
'''

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set and the test set.

results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
regularization_strengths = [5e4, 1e5, 5e5, 1e8]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# Save the best trained softmax classifer in best_softmax.                     #
# Hint: about 10 lines of code expected
################################################################################
# train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
#             batch_size=200, verbose=False)
'''
# select best learning rate and reg
for learningrate in learning_rates:
  for regularization in regularization_strengths:
    softmax=SoftmaxClassifier()

    softmax.train(X_train,y_train,learningrate,reg=regularization, num_iters=4000,batch_size=400, verbose=True)
    y_pred_val=softmax.predict(X_val)
    current_val = np.mean(y_pred_val==y_val)
    if(current_val>best_val):
      best_val = current_val
      best_softmax = softmax
      best_learning_rate = learningrate
      best_reg = regularization

print "best learning rate is ", best_learning_rate
print "best reg is ", best_reg
'''



# select best batch size and iterations
best_learning_rate = 5e-06
best_reg = 100000.0
batch_sizes = [200, 300, 400, 500]
iterations = [2000,3000,4000,5000]
'''
accuracies = np.zeros((len(batch_sizes), len(iterations)))
for batch_idx, batch_size in enumerate(batch_sizes):
  for it_idx, iteration in enumerate(iterations):
    softmax=SoftmaxClassifier()
    softmax.train(X_train,y_train,best_learning_rate,reg=best_reg, num_iters=iteration,batch_size=batch_size, verbose=True)
    y_pred_val=softmax.predict(X_val)
    current_val = np.mean(y_pred_val==y_val)
    accuracies[batch_idx,it_idx] = current_val
    if(current_val>best_val):
      best_val = current_val
      best_softmax = softmax
      best_batch_size = batch_size
      best_iteration = iteration

np.savetxt("accuracies.txt",accuracies,delimiter=',')


print "best batch size is ", best_batch_size
print "best iteration is ", best_iteration
'''
best_iteration = 3000
best_batch_size = 500


# select lambda using fmin train
best_softmax_fmin = None
best_val_fmin=-1
for reg in regularization_strengths:
    softmax=SoftmaxClassifier()
    softmax.train(X_train,y_train,best_learning_rate,reg=best_reg, num_iters=best_iteration,
                  batch_size=best_batch_size, verbose=True)
    y_pred_val=softmax.predict(X_val)
    current_val = np.mean(y_pred_val==y_val)
    if(current_val>best_val_fmin):
      best_val_fmin = current_val
      best_softmax_fmin = softmax
      best_reg_fmin = reg

print "best reg for fmin is ", best_reg_fmin

if best_softmax_fmin:
  y_test_pred = best_softmax_fmin.predict(X_test)
  test_accuracy = np.mean(y_test == y_test_pred)
  print 'softmax_fmin on raw pixels final test set accuracy: %f' % (test_accuracy, )



'''
# plot accuracies with respect to batch size and iteration

accuracies = np.genfromtxt("accuracies.txt", delimiter=',')

# plot batch size / iteration against accuracy
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(iterations,batch_sizes)
surf = ax.plot_surface(X, Y, accuracies, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#ax.set_zlim(0, 1.0)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

'''

################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# Evaluate the best softmax classifier on test set

if best_softmax:
  y_test_pred = best_softmax.predict(X_test)
  test_accuracy = np.mean(y_test == y_test_pred)
  print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

  print sklearn.metrics.confusion_matrix(y_test,y_test_pred)
  print sklearn.metrics.classification_report(y_test,y_test_pred)

  # Visualize the learned weights for each class

  theta = best_softmax.theta[1:,:].T # strip out the bias term
  theta = theta.reshape(10, 32, 32, 3)

  theta_min, theta_max = np.min(theta), np.max(theta)

  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  for i in xrange(10):
    plt.subplot(2, 5, i + 1)
  
    # Rescale the weights to be between 0 and 255
    thetaimg = 255.0 * (theta[i].squeeze() - theta_min) / (theta_max - theta_min)
    plt.imshow(thetaimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])


  plt.savefig('cifar_theta.pdf')
  plt.close()
