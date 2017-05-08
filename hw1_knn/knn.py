import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree
from scipy.stats import mode

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k):
        """
        Creates a kNN instance
        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._x = x
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).
        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        mode = [];
        item_indices_appear = dict((a,item_indices.count(a)) for a in item_indices);
        for k,v in item_indices_appear.items():
            if v == max(item_indices_appear.values()):
                mode.append(k);
        
        if len(mode) == 1:
            return mode[0];
        else:
            item_indices.sort(reverse = True)
            half = len(item_indices) // 2
            return (item_indices[half] + item_indices[~half]) / 2
		#self._y[item_indices[0]]
		
		
    def classify(self, example):
        """
        Given an example, classify the example.
        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 
        
        knn_list = []
        max_index = -1
        max_dist = 0
        
        for i in range(self._k) :
            label = self._y[i]
            train_vec=self._x[i]
            
            dist = numpy.linalg.norm(train_vec - example )
            knn_list.append((dist, label))
            
        for i in range(self._k,len(self._y)) :
			label = self._y[i]
			train_vec = self._x[i]
			
			dist=numpy.linalg.norm(train_vec - example)
			
			if max_index<0 :
				for j in range(self._k) :
					if max_dist < knn_list[j][0]:
						max_index = j
						max_dist = knn_list[max_index][0]
			
			if dist < max_dist :
				knn_list[max_index] = (dist, label)
				max_index = -1
				max_dist = 0
					
        #print knn_list
        indice_list = [x[1] for x in knn_list]
        #print indice_list
        final_label = self.majority(indice_list)
        #print final_label
        
        return final_label

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.
        :param test_x: Test data representation
        :param test_y: Test data answers
        """
     
        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        
        d=defaultdict(dict)
        k = [[0 for col in range(10)] for row in range(10)]
        #rr = []
        #tt = []
               
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
            rr = knn.classify(xx)    
            k[yy][rr] += 1
        
        #print d
        #rr = knn.classify(test_x[1])
        #yy = test_y[1]
        for ii in range(10):
		    for jj in range(10):
		        d[ii][jj] = k[ii][jj]
        
        
        #print d
        
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0
	
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=5,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    
    # You should not have to modify any of this code
    
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")
    #print knn.getmedian([1,2,3,4,5])
    #print knn.getmode([1,1,3,4,5])
    #print knn.majority([1,2,3])
    #print data.test_y[1]
    #knn.classify(data.test_x[2])
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))
