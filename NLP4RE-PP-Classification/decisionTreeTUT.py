from nltk.corpus import names
import nltk
from nltk.classify import apply_features
# nltk.download('names')
import random


# AUC https://datascience.stackexchange.com/questions/806/advantages-of-auc-vs-standard-accuracy#807
# https://www.nltk.org/book/ch06.html#sec-decision-trees

# returns a dictionary [FEATURE SET] containing relevant information about a given name
def gender_features(word):
	features = {}
	features['suffix1'] = word[-1:]
	features['suffix2'] = word[-2:]
	return features


# returns a dictionary [FEATURE SET] containing relevant information about a given name
def gender_features2(word):
	features = {}
	features["first_letter"] = word[0].lower()
	features["last_letter"] = word[-1].lower()
	features["length"] = len(word)
	return features


if __name__ == '__main__':
	gfeature = gender_features('Shrek')
	print(gfeature)
	
	# prepare a list of examples and corresponding class labels.
	labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
						[(name, 'female') for name in names.words('female.txt')])
	
	random.shuffle(labeled_names)
	print(labeled_names)
	
	# use the feature extractor to process the names data, and divide the resulting list of feature sets
	# into a training set and a test set. The training set is used to train a new "naive Bayes" classifier.
	featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
	train_set, test_set = featuresets[500:], featuresets[:500]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	
	# classify a new name, e.g. Neo
	classifier.classify(gender_features('Neo'))
	
	# examine classifier on the test set
	print(nltk.classify.accuracy(classifier, test_set))

	# examine most informative features (likelihood ratios)
	classifier.show_most_informative_features(5)
	
	# classify with new features
	# featuresets2 = [(gender_features2(n), gender) for (n, gender) in labeled_names]
	# train_set2, test_set2 = featuresets2[500:], featuresets2[:500]
	train_set2 = apply_features(gender_features2, labeled_names[500:])
	test_set2 = apply_features(gender_features2, labeled_names[:500])
	classifier2 = nltk.NaiveBayesClassifier.train(train_set2)
	
	# examine classifier on the test set
	print(nltk.classify.accuracy(classifier2, test_set2))
	
	# examine most informative features (likelihood ratios)
	classifier2.show_most_informative_features(5)
	
	# error analysis to refine feature set
	train_names = labeled_names[1500:]
	devtest_names = labeled_names[500:1500]
	test_names = labeled_names[:500]
	
	# train odel using training set and then run it on dev-test set
	train_set = [(gender_features(n), gender) for (n, gender) in train_names]
	devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
	test_set = [(gender_features(n), gender) for (n, gender) in test_names]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print(nltk.classify.accuracy(classifier, devtest_set))
	
	# # using dev-test set, we can generate a list of the errors that the classifier makes when predicting name genders
	# errors = []
	# for (name, tag) in devtest_names:
	# 	guess = classifier.classify(gender_features(name))
	# 	if guess != tag:
	# 		errors.append((tag, guess, name))
	#
	# # examine individual error cases
	# for (tag, guess, name) in sorted(errors):
	# 	print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))