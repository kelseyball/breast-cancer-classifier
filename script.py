import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# data exploration
breast_cancer = load_breast_cancer()
print "\n---------------- Data Exploration ------------------------"
print "dataset dimensions: ", breast_cancer.data.shape
print "first datapoint: ", breast_cancer.data[0]
print "feature names: ", breast_cancer.feature_names
print "target: ", breast_cancer.target
print "classes: ", breast_cancer.target_names

# split into training and test sets
print "\n---------------- Train Test Split ------------------------"
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size = 0.2, train_size=0.8, random_state=50)
print "length training data: %d, length training labels: %d" % (len(X_train), len(y_train))

print "\n---------------- Training the Classifier ------------------------"
# training the classifier with different values of k
best_k = best_score = 0
accuracies = []
for k in xrange(1, 100):
	classifier = KNeighborsClassifier(n_neighbors=k)
	classifier.fit(X_train, y_train)
	score = classifier.score(X_test, y_test)
	accuracies.append(score)
	# print "accuracy (k = %d) : %f" % (k, score)
	if score > best_score:
		best_score = score
		best_k = k
print "best value of k: %d accuracy: %f" % (best_k, best_score)

# graph results
k_list = range(1, 100) 
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()