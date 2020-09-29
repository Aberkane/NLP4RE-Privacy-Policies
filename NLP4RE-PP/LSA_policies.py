import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import seaborn as sns
from nltk.stem.wordnet import WordNetLemmatizer
import collections
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
sns.set_style('whitegrid')
stop_words = stopwords.words('english')
pd.set_option("display.max_colwidth", 200)

'''
Author: Abdel-Jaouad Aberkane, Ghent University

- This class consists of methods that focus on topic modeling using Latent Semantic Analysis (LSA)
- The result is a number of topics and for each of them a list of most frequent occuring words

Sources:
# https://towardsdatascience.com/latent-semantic-analysis-intuition-math-implementation-a194aff870f8
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/
'''


class LSAPolicies:
	
	# arg is list of strings. it removes punctuation and converts all letters to lowercase
	@staticmethod
	def pre_processing(policies_dataframe):
		counter = 0
		lemma = WordNetLemmatizer()
		# Remove punctuation
		# print("removing punctuation ...")
		policies_dataframe = [re.sub('[,\.!?&“”():*;"]', '', x) for x in policies_dataframe]
		policies_dataframe = [re.sub('[\'’/]', '', x) for x in policies_dataframe]
		
		# Convert the titles to lowercase
		policies_dataframe = [x.lower() for x in policies_dataframe]
		
		# remove short words
		policies_dataframe = [' '.join([w for w in x.split() if len(w) > 3]) for x in policies_dataframe]
		
		# tokenizing
		tokenized_policies = [x.split() for x in policies_dataframe]
		
		# lemmatize
		tokenized_policies = [[lemma.lemmatize(y) for y in x] for x in tokenized_policies]
		
		# stemming
		# porter = PorterStemmer()
		# tokenized_policies = [[porter.stem(y) for y in x] for x in tokenized_policies]
		
		# remove emailadresses and URLs
		# tokenized_policies = [[y for y in x if("@" not in y)] for x in tokenized_policies]
		tokenized_policies = [[y for y in x if (("@" not in y) and ("http" not in y) and
		                                        ("wwww" not in y) and ("com" not in y))] for x in tokenized_policies]
		
		LSAPolicies.plot_10_most_common_ngrams(tokenized_policies, 2)
		
		# remove stop-words
		# new_stop_words = ["shutterstock", "shutterstockcom", "scribd", "oracle"]
		new_stop_words = ["shutterstock", "shutterstockcom", "scribd", "oracle", "privacy", "policy", "site",
		                  "information", "unity", "frontier", "party", "service", "your", "template", "fedex",
		                  "edit", "navigation", "generator", "verizon", "doe"]
		stop_words.extend(new_stop_words)
		tokenized_policies = [[y for y in x if not y in stop_words] for x in tokenized_policies]
		
		# cooky_counter = 0
		# oracle_counter = 0
		# shutterstock_counter = 0
		# scribd_counter = 0
		# for ele in tokenized_policies:
		# 	for j in ele:
		# 		if ("oracle" in j):
		# 			oracle_counter = oracle_counter + 1
		# 			print("oracle counter: " + str(oracle_counter) + " in word: " + j)
		# 		# elif("cooky" in j):
		# 		# 	cooky_counter = cooky_counter + 1
		# 		# 	print("cooky counter: " + str(cooky_counter) + " in word: " + j)
		# 		elif ("shutterstock" in j):
		# 			shutterstock_counter = shutterstock_counter + 1
		# 			print("shutterstock counter: " + str(shutterstock_counter) + " in word: " + j)
		# 		elif ("scribd" in j):
		# 			scribd_counter = scribd_counter + 1
		# 			print("scribd counter: " + str(scribd_counter) + " in word: " + j)
		#
		
		# de-tokenization
		# print("de-tokenization ...")
		
		detokenized_policies = []
		for i in range(len(policies_dataframe)):
			t = ' '.join(tokenized_policies[i])
			detokenized_policies.append(t)
		LSAPolicies.plot_wordcloud(detokenized_policies)
		return detokenized_policies
	
	@staticmethod
	def count_TFIDF_vectorizer(policies, topics, tfidf=False):
		if tfidf:
			print("Topic modeling using TFIDF-vectorizer ...")
			vectorizer = TfidfVectorizer(stop_words='english',
			                             # max_features=50,  # keep top 1000 terms
			                             # max_df=0.5,
			                             ngram_range=(1, 3),
			                             smooth_idf=True)
			
			# only store elements that have a score (that do occur in multiple documents)
			X = vectorizer.fit_transform(policies)
		else:
			print("count-vectorizer")
			df_policies = pd.DataFrame(policies)
			
			# Initialise the count vectorizer with the English stop words
			vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
			
			# Fit and transform the processed titles
			X = vectorizer.fit_transform(policies)
		
		# LSAPolicies.plot_10_most_common_words(X, vectorizer)
		# check shape of the document-term matrix: 319 docs and 524855 terms
		# print(X.shape)
		
		# decompose matrix into 3 matrices: USV
		# U: m x k matrix, rows are documents and columns are 'mathematical concepts'
		# S: k x k diagonal matrix, elements are amount of variation captured from each concepts
		# V: m x k matrix (transposed), rows are terms and columns are concepts
		
		# SVD represent documents and terms in vectors
		lsa = TruncatedSVD(n_components=topics, algorithm='randomized', n_iter=100)
		lsa.fit(X)
		
		# lsa_topic_matrix = lsa.fit_transform(X)
		# lsa_keys = get_keys(lsa_topic_matrix)
		# lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
		
		# this is the first row of V
		# print((lsa.components_[0]))
		
		# get terms in same order as lsa.components_[0]
		terms = vectorizer.get_feature_names()
		LSAPolicies.print_topics(lsa, terms)
	
	@staticmethod
	def print_topics(lsa, terms):
		for i, comp in enumerate(lsa.components_):
			terms_comp = zip(terms, comp)
			sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:5]
			print("Topic " + str(i) + ": ")
			for t in sorted_terms:
				print(t[0])
			print(" ")
	
	@staticmethod
	def plot_10_most_common_ngrams(tokenized_policies, n):
		import matplotlib.pyplot as plt
		# words = count_vectorizer.get_feature_names()
		# total_counts = np.zeros(len(words))
		# for t in count_data:
		# 	total_counts += t.toarray()[0]
		tokenized_policies_flat = [item for sublist in tokenized_policies for item in sublist]
		n_grams_policies = ngrams(tokenized_policies_flat, n)
		bigrams_policies = collections.Counter(n_grams_policies)
		print(bigrams_policies.most_common(10))
		
		count_dict = bigrams_policies.most_common()
		count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
		words = [w[0] for w in count_dict]
		counts = [w[1] for w in count_dict]
		x_pos = np.arange(len(words))
		
		# print(type(words[0]))
		words_list = [','.join(x) for x in words]
		words_list = [re.sub('[\.!?&“”\'():*;"]', '', x) for x in words_list]
		# print(words_list[0])
		# exit(0)
		
		# plt.figure(2, figsize=(15, 15 / 1.6180))
		ngrams_title = str(n) + '-grams'
		title = '10 most common ' + ngrams_title
		plt.figure(2, figsize=(30, 30 / 1.6180))
		plt.subplot(title=title)
		sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
		sns.barplot(x_pos, counts, palette='husl')
		plt.tick_params(axis='x', labelsize=8)
		plt.xticks(x_pos, words_list)
		plt.xlabel(ngrams_title)
		plt.ylabel('counts')
		plt.show()
	
	@staticmethod
	def plot_10_most_common_words(count_data, count_vectorizer):
		import matplotlib.pyplot as plt
		words = count_vectorizer.get_feature_names()
		total_counts = np.zeros(len(words))
		for t in count_data:
			total_counts += t.toarray()[0]
		
		count_dict = (zip(words, total_counts))
		count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
		words = [w[0] for w in count_dict]
		counts = [w[1] for w in count_dict]
		x_pos = np.arange(len(words))
		
		plt.figure(2, figsize=(15, 15 / 1.6180))
		plt.subplot(title='10 most common words')
		sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
		sns.barplot(x_pos, counts, palette='husl')
		plt.xticks(x_pos, words, rotation=90)
		plt.xlabel('words')
		plt.ylabel('counts')
		plt.show()
	
	@staticmethod
	def plot_wordcloud(policies):
		# Join the different processed titles together.
		long_string = ','.join(policies)
		
		# Create a WordCloud object
		wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
		
		# Generate a word cloud
		wordcloud.generate(long_string)
		
		# Visualize the word cloud
		# make figure to plot
		plt.figure()
		plt.imshow(wordcloud, interpolation="bilinear")
		
		# remove axes
		plt.axis("off")
		# show the result
		plt.show()
