import pandas as pd
import numpy as np
from collections import Counter
from pdb import set_trace
class my_DT:

	def __init__(self, criterion="gini", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
		# criterion = {"gini", "entropy"},
		# Stop training if depth = max_depth
		# Only split node if impurity decrease >= min_impurity_decrease after the split
		#   Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
		# Only split node with >= min_samples_split samples
		self.criterion = criterion
		self.max_depth = int(max_depth)
		self.min_impurity_decrease = min_impurity_decrease
		self.min_samples_split = int(min_samples_split)
		self.tree = {}

	def impurity(self, labels):
		# Calculate impurity
		# Input is a list of labels
		# Output impurity score <= 1
		stats = Counter(labels)
		N = float(len(labels))
		if self.criterion == "gini":
			# Implement gini impurity
			impure = 1
			for key in stats:
				impure -= (stats[key]/N)**2
		elif self.criterion == "entropy":
			# Implement entropy impurity
			impure = 0
			for key in stats:
				impure -= (stats[key]/N)*np.log2(stats[key]/N)
		else:
			raise Exception("Unknown criterion.")
		return impure

	def find_best_split(self, pop, X, labels):
		# Find the best split
		# Inputs:
		#   pop:    indices of data in the node
		#   X:      independent variables of training data
		#   labels: dependent variables of training data
		# Output: tuple(best feature to split, weighted impurity score of best split, splitting point of the feature, [indices of data in left node, indices of data in right node], [weighted impurity score of left node, weighted impurity score of right node])
		######################
		best_feature = None
		for feature in X.keys():
			cans = np.array(X[feature][pop])
			cans_sorted = np.argsort(cans)
			n = len(cans_sorted)
			impures = []
			impure = []
			for i in range(n - 1):
				if cans[cans_sorted[i]] == cans[cans_sorted[i + 1]]:
					impure.append(np.inf)
					impures.append([])
				else:
					impures.append([self.impurity(labels[pop[cans_sorted[:i + 1]]]) * (i + 1),
									(n - i - 1) * self.impurity(labels[pop[cans_sorted[i + 1:]]])])
					impure.append(np.sum(impures[-1]))

			min_impure = np.min(impure)

			if min_impure < np.inf and (best_feature == None or best_feature[1] > min_impure):
				split = np.argmin(impure)
				best_feature = (feature, min_impure, (cans[cans_sorted][split] + cans[cans_sorted][split + 1]) / 2.0,
								[pop[cans_sorted[:split + 1]], pop[cans_sorted[split + 1:]]], impures[split])

		return best_feature

	def fit(self, X, y):
		# X: pd.DataFrame, independent variables, float
		# y: list, np.array or pd.Series, dependent variables, int or str
		self.classes_ = list(set(list(y)))

		labels = np.array(y)
		N = len(y)
		population = {0: np.array(range(N))}
		impurity = {0: self.impurity(labels[population[0]])*N}
		level = 0
		nodes = [0]
		while level < self.max_depth and nodes:
			next_nodes = []
			for node in nodes:
				current_pop = population[node]
				current_impure = impurity[node]
				if len(current_pop) < self.min_samples_split or current_impure == 0 or level+1 == self.max_depth:
					self.tree[node] = Counter(labels[current_pop])
				else:
					best_feature = self.find_best_split(current_pop, X, labels)
					if best_feature and (current_impure - best_feature[1])>self.min_impurity_decrease*N:
						self.tree[node]=(best_feature[0], best_feature[2])
						next_nodes.extend([node*2+1,node*2+2])
						population[node*2+1] = best_feature[3][0]
						population[node*2+2] = best_feature[3][1]
						impurity[node*2+1] = best_feature[4][0]
						impurity[node*2+2] = best_feature[4][1]
					else:
						self.tree[node] = Counter(labels[current_pop])
			nodes = next_nodes
			level += 1
		return

	def predict(self, X):
		# X: pd.DataFrame, independent variables, float
		# return predictions: list
		# write your code below

		predictions = []
		for i in range(len(X)):
			node = 0
			while True:

				if type(self.tree[node])==Counter:
					label = list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
					predictions.append(label)
					break
				else:
					if X[self.tree[node][0]][i] < self.tree[node][1]:
						node = node*2+1
					else:
						node = node*2+2

		return predictions

	def predict_proba(self, X):
		# X: pd.DataFrame, independent variables, float
		# Eample:
		# self.classes_ = {"2", "1"}
		# the reached node for the test data point has {"1":2, "2":1}
		# then the prob for that data point is {"2": 1/3, "1": 2/3}
		# return probs = pd.DataFrame(list of prob, columns = self.classes_)
		# write your code below

		predictions = []
		for i in range(len(X)):
			node = 0
			while True:
				if type(self.tree[node])==Counter:
					N = float(np.sum(list(self.tree[node].values())))
					predictions.append({key: self.tree[node][key]/N for key in self.classes_})
					break
				else:
					if X[self.tree[node][0]][i] < self.tree[node][1]:
						node = node*2+1
					else:
						node = node*2+2
		probs = pd.DataFrame(predictions, columns = self.classes_)
		return probs



