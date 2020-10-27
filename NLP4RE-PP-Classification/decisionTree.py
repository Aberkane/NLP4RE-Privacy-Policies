# from nltk.corpus import movie_reviews
import numpy as np
import ssl
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from langdetect import detect
import os.path
from nltk.stem.wordnet import WordNetLemmatizer
# from http.client import IncompleteRead


import re
# from urllib.request import urlopen, Request
# from urllib.error import URLError, HTTPError
from newspaper import Article
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from http.client import IncompleteRead

import socket
import random
# nltk.download('brown')
stop_words = stopwords.words('english')
# from nltk.corpus import brown


# Sources: https://subscription.packtpub.com/book/application_development/9781782167853/7/ch07lvl1sec78/training-a-decision-tree-classifier
# https://www.nltk.org/book/ch06.html#sec-decision-trees

# define a feature extractor for documents, so the classifier will know which aspects of the data it should pay
# attention to
# def document_features(document):
# 	document_words = set(document) # looking up in a set is faster than in list
# 	features = {}
# 	for word in word_features:
# 		features['contains({})'.format(word)] = (word in document_words)
# 	return features

def list_to_string(s):
	# initialize an empty string
	str1 = ""
	
	# traverse in the string
	for ele in s:
		str1 += ele
	
	# return string
	return str1


def scrape_policy_url(policy_url):
	parsed_policy = ""
	try:
		article = Article(policy_url)
		article.download()  # Downloads the link’s HTML content
		article.parse()  # Parse the article
		article.nlp()
		# print(policy_url)
		# print(article.text)
		parsed_policy = article
	# except article.ArticleException as ae:
	# print(ae)
	except:
		pass
	# avoid too many requests error from Google
	
	return parsed_policy


def find_relevant_paragraph(policy_url):
	# policy_url = "https://irs.gov/privacy-disclosure/report-phishing"
	
	pre_dataframe = [[]]
	potential_paragraph = ""
	headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) "
	                         "Chrome/41.0.2228.0 Safari/537.3"}
	
	# header = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0', }
	
	try:
		req = Request(policy_url, headers=headers)
		html = urlopen(req, timeout=1)
		
		# fill first element of the list (first cell of row) with the relevant website title
		# policy_list.append(urlparse(policy.url).netloc + ".txt")
		# html = urlopen('https://automattic.com/privacy/')
		
		bs = BeautifulSoup(html, "html.parser")
		# bs = BeautifulSoup(req.text, "lxml")
		
		# 6 levels of HTML-headers
		# titles = bs.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
		# print('List all the header tags :', *titles, sep='\n\n')
		
		for header in bs.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
			# print(header.text.strip())
			if ("right" or "rights") in header.text.strip().lower():
				
				nextNode = header
				while True:
					nextNode = nextNode.nextSibling
					if nextNode is None:
						break
					if isinstance(nextNode, NavigableString):
						# print(nextNode.strip())
						potential_paragraph = potential_paragraph + " " + nextNode.strip()
					if isinstance(nextNode, Tag):
						if nextNode.name == "h2":
							break
						# print(nextNode.get_text(strip=True).strip())
						potential_paragraph = potential_paragraph + " " + nextNode.get_text(strip=True).strip()
	
	# policy_list.append(header.text.strip())
	# print("*** *** *** END")
	# pre_dataframe.append(policy_list)
	except HTTPError as e:
		# do something
		print('Error code: ', e.code)
		pass
	except URLError as e:
		# do something
		print('Reason: ', e.reason)
		pass
	except socket.timeout as e:
		# if urlopen takes too long just skip the url
		pass
	except IncompleteRead:
		# Oh well, reconnect and keep trucking
		pass
	except ConnectionResetError:
		pass
	except ssl.SSLWantReadError:
		pass
	
	return potential_paragraph


def reduce_policy(policy_url):
	potential_reduced_policy = find_relevant_paragraph(policy_url)
	return potential_reduced_policy


# check language of policy and verify whether it can be reduced to a smaller size: if so, change
def read_texts(n_words_policy, check_reduced):
	# print("Extracting texts from: " + data_dir)
	all_files = os.listdir(data_dir)
	selected_files = []
	policies_list = []
	titles = []
	
	# filter on txt files (redundant, but safety first)
	txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
	# print("Potential policies found: " + str(len(all_files)))
	
	for file in txt_files:
		# print(file)
		new_file = []
		potential_policy_text = ""
		with open((data_dir + "\\" + file), "r", encoding="utf8") as policy:
			data = policy.readlines()
			policy_text = list_to_string(data[5:])
			policy_url = list_to_string(data[1])[5:]
			new_file[:5] = data[:5]
			if len(policy_text.split()) > n_words_policy:
				lang = detect(policy_text)
				if lang == 'en':
					# print("het is engels")
					# print(lang)
					
					# -if text can be reduced, reduce
					if check_reduced:
						potential_policy_text = reduce_policy(policy_url)
					
					if len(potential_policy_text) > 20:
						# print("Considering... reduced text")
						policies_list.append(potential_policy_text)
						new_file.append("Reduced Policy: " + potential_policy_text)
					else:
						# print("Considering... full text")
						# otherwise consider text in full
						policies_list.append(policy_text)
					title = (data[0].rstrip("\n").replace("Title: ", ""))
					titles.append(title)
					selected_files.append(file)
				else:
					# print("het is geen engels")
					# print(lang)
					pass
		if len(potential_policy_text) > 20:
			with open((data_dir + "\\" + file), "w", encoding="utf8") as selected_file:
				selected_file.writelines(new_file)
	
	print("Final number of policies after filtering: " + str(len(policies_list)))
	# return policies_list, titles
	return selected_files, policies_list


# link policies from policy database to the urls in the scoped dataframe
def select_policies(url_list, policy_list, policy_list_text):
	url_policy_list = []
	counter = 0
	for url in url_list:
		url_counter = 0
		for i, policy in enumerate(policy_list, start=0):  # default is zero
			if url in policy:
				if url_counter == 0:
					url_policy_list.append(policy_list_text[i])
					# print(url)
					# print(policy)
				else:
					# print("Dubbel")
					# if there are multiple policies for one website, select the one that is longer
					if len(policy_list_text[i]) > len(url_policy_list[-1]):
						url_policy_list[-1] = policy_list_text[i]
				url_counter = 1
		if url_counter == 0:
			url_policy_list.append("")
	
	return url_policy_list

#
# def pos_features(word):
# 	features = {}
# 	for suffix in common_suffixes:
# 		features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
# 	return features


def read_data():
	df_pp_read = pd.read_excel(dir)
	
	# SELECT ONLY URLS THAT ARE EITHER RELEVANT OR NOT RELEVANT WITH EXCLUSION CRITERIA CODE E1 (NOT RELEVANT)
	df_pp_relevant = df_pp_read[(df_pp_read['included'] == 'y') | (df_pp_read['included'] == 'e1')]
	
	df_pp_relevant_scoped = df_pp_relevant[
		['URL', 'included', 'UR_explicitly_mentioned', 'access', 'rectification', 'erasure', 'restriction',
		 'data_portability', 'object', 'automated_decision_making']]
	
	df_user_rights = df_pp_relevant[['access', 'rectification', 'erasure', 'restriction',
	                                 'data_portability', 'object', 'automated_decision_making']]
	
	# plot_stats(df_user_rights)
	
	# LIST OF URLs
	url_list = df_pp_relevant_scoped['URL'].tolist()
	
	return df_pp_read, df_pp_relevant, df_pp_relevant_scoped, df_user_rights, url_list


# def document_features(document):
# 	document_words = set(document)  # faster than list
# 	features = {}
# 	for word in word_features:
# 		features['contains({})'.format(word)] = (word in document_words)
# 	return features


def document_features2(document):
	document_words = set(document)  # faster than list
	features = {}
	for word in word_features_p:
		features['contains({})'.format(word)] = (word in document_words)
	return features


def pre_processing(policies_list):
	counter = 0
	lemma = WordNetLemmatizer()
	# Remove punctuation
	# print("removing punctuation ...")
	policies_list = [re.sub('[,\.!?&“”():*;"]', '', x) for x in policies_list]
	policies_list = [re.sub('[\'’/]', '', x) for x in policies_list]
	
	# Convert the titles to lowercase
	policies_list = [x.lower() for x in policies_list]
	
	# remove short words
	policies_list = [' '.join([w for w in x.split() if len(w) > 3]) for x in policies_list]
	
	# tokenizing
	tokenized_policies = [x.split() for x in policies_list]
	
	# lemmatize
	tokenized_policies = [[lemma.lemmatize(y) for y in x] for x in tokenized_policies]
	
	# stemming
	# porter = PorterStemmer()
	# tokenized_policies = [[porter.stem(y) for y in x] for x in tokenized_policies]
	
	# remove emailadresses and URLs
	# tokenized_policies = [[y for y in x if("@" not in y)] for x in tokenized_policies]
	tokenized_policies = [[y for y in x if (("@" not in y) and ("http" not in y) and
	                                        ("wwww" not in y) and ("com" not in y))] for x in tokenized_policies]
	
	
	# remove stopwords
	tokenized_policies = [[y for y in x if not y in stop_words] for x in tokenized_policies]
	
	# print(relevant_words_lemmatized)
	# print(tokenized_policies)
	# print("length of policy: " + str(len(tokenized_policies)))
	# print("\n")
	
	detokenized_policies = []
	for i in range(len(policies_list)):
		t = ' '.join(tokenized_policies[i])
		detokenized_policies.append(t)
		
	return detokenized_policies


def correctNAN(df_pp_tbc):
	df_pp_tbc.UR_explicitly_mentioned = pd.to_numeric(df_pp_tbc.UR_explicitly_mentioned, errors='coerce').fillna(0).astype(
		np.int64)
	df_pp_tbc.access = pd.to_numeric(df_pp_tbc.access, errors='coerce').fillna(0).astype(np.int64)
	df_pp_tbc.rectification = pd.to_numeric(df_pp_tbc.rectification, errors='coerce').fillna(0).astype(np.int64)
	df_pp_tbc.erasure = pd.to_numeric(df_pp_tbc.erasure, errors='coerce').fillna(0).astype(np.int64)
	df_pp_tbc.restriction = pd.to_numeric(df_pp_tbc.restriction, errors='coerce').fillna(0).astype(np.int64)
	df_pp_tbc.data_portability = pd.to_numeric(df_pp_tbc.data_portability, errors='coerce').fillna(0).astype(np.int64)
	df_pp_tbc.object = pd.to_numeric(df_pp_tbc.object, errors='coerce').fillna(0).astype(np.int64)
	df_pp_tbc.automated_decision_making = pd.to_numeric(df_pp_tbc.automated_decision_making, errors='coerce').fillna(0).astype(
		np.int64)
	return df_pp_tbc


if __name__ == '__main__':
	pd.set_option("display.max_rows", None, "display.max_columns", None)  # print all values of dataframe
	dir = "data/PP_comparison_classification.xlsx"
	data_dir = "data/privacy_policies"
	
	print("Reading data from: " + dir + " ...")
	df_pp_read, df_pp_relevant, df_pp_relevant_scoped, df_user_rights, url_list = read_data()
	
	# search for corresponding policies
	print("Extracting policy texts from: " + data_dir + " ...")
	policy_list, policy_list_text = read_texts(140, False)
	
	print("Link dataset to extracted policies ...")
	url_list_policies = select_policies(url_list, policy_list, policy_list_text)
	url_list_policies = pre_processing(url_list_policies)
	
	# insert column with the corresponding privacy policies text
	df_pp_relevant_scoped.insert(2, 'policy', url_list_policies)
	# df_pp_relevant.insert(2, 'policy', url_list_policies)
	
	# select df where policies are empty
	df_pp_relevant_scoped[(df_pp_relevant_scoped['policy'] == "")]
	
	categories = ['UR_explicitly_mentioned', 'access', 'rectification', 'erasure', 'restriction',
	              'data_portability', 'object', 'automated_decision_making']
	
	# what is the policy distribution?
	df_pp_filtered = df_pp_relevant_scoped[(df_pp_relevant_scoped['policy'] != "")]
	print("Policy distribution after filtering out empty policy texts")
	print("* Number of policies included with label \"y\": " + str(len(df_pp_filtered[(df_pp_filtered['included'] == 'y')])))
	print("* Number of policies excluded with label \"e1\": " + str(len(df_pp_filtered[(df_pp_filtered['included'] == 'e1')])))
	print("Total number of policies: " + str(len(df_pp_filtered)))
	for category in categories:
		print("Category {} - positive labels: {}".format(category, len(df_pp_filtered[(df_pp_filtered[category] == 1)])))
		print("Category {} - negative labels: {} \n".format(category, len(df_pp_filtered[(df_pp_filtered[category] == 0)])))
	
	# df_test = df_pp_relevant[(df_pp_relevant['policy'] != "")]

	# fill NaN with 0
	# df_pp = correctNAN(df_pp_filtered)
	df_pp = df_pp_filtered
	
	# check if there are other entries than 0 or 1
	df_pp[(df_pp['UR_explicitly_mentioned'] != 0) & (df_pp['UR_explicitly_mentioned'] != 1)]
	
	# split data to train and test set


	policy_dict = [(policy_t.split(), label)
	             for policy_t in df_pp['policy'].to_list()
	             for label in df_pp['object'].to_list()]
	
	# documents = [(list(movie_reviews.words(fileid)), category)
	#              for category in movie_reviews.categories()
	#              for fileid in movie_reviews.fileids(category)]

	# random.shuffle(documents)
	random.shuffle(policy_dict)
	
	#
	# # To limit the number of features that the classifier needs to process, we begin by constructing a list of the 2000
	# # most frequent words in the overall corpus [1]. We can then define a feature extractor [2] that simply checks whether
	# # each of these words is present in a given document.
	tokenized_policies = [x.split() for x in df_pp['policy'].to_list()]
	#flatten
	tokenized_policies_f = [word for policy in tokenized_policies for word in policy]
	all_words_p = nltk.FreqDist(w.lower() for w in tokenized_policies_f)
	word_features_p = list(all_words_p)[:200]
	
	# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
	# word_features = list(all_words)[:2000]
	#
	# featuresets = [(document_features(d), c) for (d, c) in documents]
	# train_set, test_set = featuresets[100:], featuresets[:100]
	# classifier = nltk.NaiveBayesClassifier.train(train_set)
	
	featuresets_p = [(document_features2(d), c) for (d, c) in policy_dict]
	
	# featuresets = [(document_features(d), c) for (d, c) in documents]
	# train_set, test_set = featuresets[100:], featuresets[:100]
	
	# size = int(len(featuresets) * 0.1)
	size_p = int(len(featuresets_p) * 0.4)
	# train_set, test_set = featuresets[size:], featuresets[:size]
	train_set_p, test_set_p = featuresets_p[size_p:], featuresets_p[:size_p]
	
	classifier = nltk.DecisionTreeClassifier.train(train_set_p)
	print(nltk.classify.accuracy(classifier, test_set_p))
	print(classifier.pseudocode(depth=4))
	classifier.show_most_informative_features(5)
