import argparse
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
import matplotlib.pyplot as plt
import math

np.random.seed(1234)

class FoursAndNines:
	"""
	Class to store MNIST data
	"""

	def __init__(self, location):

		import cPickle, gzip

		# Load the dataset
		f = gzip.open(location, 'rb')

		# Split the data set 
		train_set, valid_set, test_set = cPickle.load(f)

		# Extract only 4's and 9's for training set 
		self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
		self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
		self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])
		
		# Shuffle the training data 
		shuff = np.arange(self.x_train.shape[0])
		np.random.shuffle(shuff)
		self.x_train = self.x_train[shuff,:]
		self.y_train = self.y_train[shuff]

		# Extract only 4's and 9's for validation set 
		self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
		self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
		self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])
		
		# Extract only 4's and 9's for test set 
		self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
		self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
		self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])
		
		f.close()

class AdaBoost:
	def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1)):
		"""
		Create a new adaboost classifier.
		
		Args:
			n_learners (int, optional): Number of weak learners in classifier.
			base (BaseEstimator, optional): Your general weak learner 
		Attributes:
			base (estimator): Your general weak learner 
			n_learners (int): Number of weak learners in classifier.
			alpha (ndarray): Coefficients on weak learners. 
			learners (list): List of weak learner instances. 
		"""
		
		self.n_learners = n_learners 
		self.base = base
		self.alpha = np.zeros(self.n_learners)
		self.learners = []
		
	def fit(self, X_train, y_train):
		"""
		Train AdaBoost classifier on data. Sets alphas and learners. 
		
		Args:
			X_train (ndarray): [n_samples x n_features] ndarray of training data   
			y_train (ndarray): [n_samples] ndarray of data 
		"""

		# TODO 

		# Hint: You can create and train a new instantiation 
		# of your sklearn weak learner as follows 
		
		w = np.ones(len(y_train))
		
		for k in range(self.n_learners) :
			h = clone(self.base)
			h.fit(X_train, y_train, sample_weight=w)
			y_test = h.predict(X_train)
			numerator = 0.0
			denominator = 0.0
			for i in range(len(y_train)) :
				if y_train[i] != y_test[i] :
					numerator += w[i] 
				denominator += w[i]
			error_rate = numerator / denominator
			#print error_rate
			self.alpha[k] = 0.5 * math.log((1-error_rate) / error_rate)
			for i in range(len(w)) :
				w[i] = w[i] * math.exp(0 - self.alpha[k] * y_train[i] * y_test[i])
			w = w / np.linalg.norm(w)
			self.learners.append(h)
		
	def predict(self, X):
		"""
		Adaboost prediction for new data X.
		
		Args:
			X (ndarray): [n_samples x n_features] ndarray of data 
			
		Returns: 
			[n_samples] ndarray of predicted labels {-1,1}
		"""

		# TODO 
		result = np.zeros(X.shape[0])
		value = 0
		for j in range(len(self.learners)) :
			value += self.alpha[j] * self.learners[j].predict(X)
		result = np.sign(value)

		return result
	
	def score(self, X, y):
		"""
		Computes prediction accuracy of classifier.  
		
		Args:
			X (ndarray): [n_samples x n_features] ndarray of data 
			y (ndarray): [n_samples] ndarray of true labels  
			
		Returns: 
			Prediction accuracy (between 0.0 and 1.0).
		"""

		# TODO 
		result = self.predict(X)
		correct_cnt = 0.0
		total_cnt = 0.0
		for i in range(len(result)) :
			total_cnt += 1.0
			if result[i] == y[i] :
				correct_cnt += 1.0
		accuracy = correct_cnt / total_cnt
		
		return accuracy
    
	def staged_score(self, X, y):
		"""
		Computes the ensemble score after each iteration of boosting 
		for monitoring purposes, such as to determine the score on a 
		test set after each boost.
		
		Args:
			X (ndarray): [n_samples x n_features] ndarray of data 
			y (ndarray): [n_samples] ndarray of true labels  
			
		Returns: 
			[n_learners] ndarray of scores 
		"""

		# TODO 
		stage_score = np.zeros(self.n_learners)
		result = self.predict(X)
		value = 0
		for j in range(len(self.learners)) :
			correct_cnt = 0.0
			total_cnt = 0.0
			value += self.alpha[j] * self.learners[j].predict(X)
			result = np.sign(value)
			for i in range(len(result)) :
				total_cnt += 1.0
				if y[i] == result[i] :
					correct_cnt += 1.0
			stage_score[j] = correct_cnt / total_cnt
		

		return  stage_score


def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
		plt.savefig(outname)
	else:
		plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
						help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
						help="Number of weak learners to use in boosting")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

	# An example of how your classifier might be called 
	clf = AdaBoost(n_learners=300, base=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
	clf.fit(data.x_train, data.y_train)
	train_stage_score = clf.staged_score(data.x_train, data.y_train)
	test_stage_score = clf.staged_score(data.x_test , data.y_test)
	print train_stage_score
	print test_stage_score
	x = np.linspace(0 , 1 , len(test_stage_score))
	train = plt.plot(train_stage_score , 'b' , label = 'train accuracy')
	test = plt.plot(test_stage_score , 'r' , label = 'test accuracy')
	plt.xlabel('iteration')
	plt.ylabel('accuracy')
	plt.legend(loc = 'lower right')
	plt.show()
