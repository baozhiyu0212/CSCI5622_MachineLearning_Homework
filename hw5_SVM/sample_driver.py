import argparse
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC 
from sklearn.grid_search import GridSearchCV

class ThreesAndEights:
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

		self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0],:]
		self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0]]

		shuff = np.arange(self.x_train.shape[0])
		np.random.shuffle(shuff)
		self.x_train = self.x_train[shuff,:]
		self.y_train = self.y_train[shuff]

		self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0],:]
		self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0]]
		
		self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0],:]
		self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0]]

		f.close()

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def plotGrid(grid):
    
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(crng), len(drng))

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('degree')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(drng)), drng)
    plt.yticks(np.arange(len(crng)), crng)
    plt.title('Validation accuracy')
    plt.show()
	
	


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

	parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	args = parser.parse_args()

	data = ThreesAndEights("../data/mnist.pkl.gz")
	if args.limit > 0:
		train_x = data.x_train[:args.limit]
		train_y = data.y_train[:args.limit]
	else:
		train_x = data.x_train
		train_y = data.y_train
        
        
	clf = SVC(kernel = "linear" , C = 0.01)
	k = clf.fit(train_x , train_y)
	score = clf.score(data.x_test , data.y_test)
	exam = clf.support_vectors_
	print score
	print exam.shape
	mnist_digit_show(exam[200,:])
	#crng = np.logspace(-2 , 10 , 13)
	#grng = np.logspace(-9, 3, 13)
	#drng = [1,2,3,4,5,6,7,8,9]
	#param_grid = dict(degree = drng , C = crng)
	
	#grid = GridSearchCV(SVC(kernel = "poly") , param_grid = param_grid , cv = 3)
	#grid.fit(train_x , train_y)
	#plotGrid(grid)
	#print grid.grid_scores_
	#print grid.best_score_
	#print grid.best_params_
	# -----------------------------------
	# Plotting Examples 
	# -----------------------------------

	# Display in on screen  
	#mnist_digit_show(data.x_train[ 0,:])

	# Plot image to file 
	#mnist_digit_show(data.x_train[1,:], "mnistfig.png")
