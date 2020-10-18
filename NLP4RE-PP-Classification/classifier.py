# %matplotlib inline

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from langdetect import detect
import os.path
from nltk.stem.wordnet import WordNetLemmatizer
from http.client import IncompleteRead

import socket
import re
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from newspaper import Article
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urlparse
import requests

stop_words = stopwords.words('english')
from sklearn.pipeline import Pipeline
import seaborn as sns

"""
Code inspired by: https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
by Susan Li.
"""

"""
Simple function to cast list to string
# https://www.geeksforgeeks.org/python-program-to-convert-a-list-to-string/
"""


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
	return potential_paragraph


def reduce_policy(policy_url):
	potential_reduced_policy = find_relevant_paragraph(policy_url)
	return potential_reduced_policy


def read_texts(dir, n_words_policy):
	print("Extracting texts from: " + dir)
	all_files = os.listdir(dir)
	selected_files = []
	policies_list = []
	titles = []
	
	# filter on txt files (redundant, but safety first)
	txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
	print("Potential policies found: " + str(len(all_files)))
	
	for file in txt_files:
		# print(file)
		with open((dir + "\\" + file), "r", encoding="utf8") as policy:
			data = policy.readlines()
			policy_text = list_to_string(data[5:])
			policy_url = list_to_string(data[1])[5:]
			print(policy_url)
			if len(policy_text.split()) > n_words_policy:
				lang = detect(policy_text)
				if lang == 'en':
					# print("het is engels")
					# print(lang)
					potential_policy_text = reduce_policy(policy_url)
					# -if text can be reduced, reduce
					if len(potential_policy_text) > 20:
						print("Considering... reduced text")
						policies_list.append(potential_policy_text)
					else:
						print("Considering... full text")
						# otherwise consider text in full
						policies_list.append(policy_text)
					title = (data[0].rstrip("\n").replace("Title: ", ""))
					titles.append(title)
					selected_files.append(file)
				else:
					# print("het is geen engels")
					# print(lang)
					pass
	
	print("Final number of policies after filtering: " + str(len(policies_list)))
	# return policies_list, titles
	return selected_files, policies_list


def plot_stats(df_user_rights):
	counts = []
	categories = list(df_user_rights.columns.values)
	for i in categories:
		counts.append((i, df_user_rights[i].sum()))
	df_stats = pd.DataFrame(counts, columns=['user_right', 'number_of_policies'])
	
	# NUMBER OF POLICIES IN EACH CATEGORY
	# df_stats.plot(x='user_right', y='number_of_policies', kind='bar', legend=False, grid=True, figsize=(10, 8))
	# plt.title("Number of policies per user right")
	# plt.ylabel('#_of_occurrences', fontsize=10)
	# plt.xlabel('user_right', fontsize=10)
	
	# NUMBER OF MULTI-LABELED POLICIES
	# rowsums = df_user_rights.iloc[:, 2:].sum(axis=1)
	# x = rowsums.value_counts()
	
	# plot
	# plt.figure(figsize=(8, 5))
	# ax = sns.barplot(x.index, x.values)
	# plt.title("Multiple user rights per privacy policy")
	# plt.ylabel('# of Occurrences', fontsize=12)
	# plt.xlabel('# of user rights', fontsize=12)
	
	# PERCENTAGE POLICIES NOT LABELED
	percentage_unlabeled = len(df_user_rights[(df_user_rights['access'] == 0) & (
			df_user_rights['rectification'] == 0) & (df_user_rights['erasure'] == 0) &
	                                          (df_user_rights['restriction'] == 0) &
	                                          (df_user_rights['data_portability'] == 0) &
	                                          (df_user_rights['object'] == 0) &
	                                          (df_user_rights['automated_decision_making'] == 0)]) / len(df_user_rights)
	print('{} % of the privacy policies is not labeled for the selected user rights'.format(
		int(percentage_unlabeled * 100)))


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
					print(url)
					print(policy)
				else:
					print("Dubbel")
					# if there are multiple policies for one website, select the one that is longer
					if len(policy_list_text[i]) > len(url_policy_list[-1]):
						url_policy_list[-1] = policy_list_text[i]
				url_counter = 1
		if url_counter == 0:
			url_policy_list.append("")
	
	return url_policy_list


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
	
	relevant_words = ["right", "access", "subject", "Information",
                      "personal", "provide", "right", "rectification",
	                  "correct", "rectify", "correction", "rectification", "data", "right", "erasure", "right",
                      "forgotten ", "delete", "erase", "erasure", "restrict", "processing", "limit",
						"processing", "general", "protection", "regulation", "europe", "EU"]
	# relevant_words_access = ['Right of access by the data subject data access Information and access to personal data provide information']
	# relevant_words_rectification = ['right to rectification correct rectify correction rectification of data']
	# relevant_words_erasure = ['right to erasure right to be forgotten delete erase erasure of data']
	# relevant_words_restriction = ['restrict processing limit processing']
	# relevant_words_data_portability = ['Right to data portability right to obtain a copy of personal data The controller shall provide a copy of '
	#                                    'the personal data undergoing processing']
	# relevant_words_data_object = ['right to object to processing withdraw consent']
	# relevant_words_data_ADM = ['automated data processing decision making decision-making marketing profiling']
	
	# remove stop-words
	# new_stop_words = ["shutterstock", "shutterstockcom", "scribd", "oracle"]
	# new_stop_words = ["shutterstock", "shutterstockcom", "scribd", "oracle", "privacy", "policy", "site",
	#                   "information", "unity", "frontier", "party", "service", "your", "template", "fedex",
	#                   "edit", "navigation", "generator", "verizon", "doe"]
	# stop_words.extend(new_stop_words)
	
	# lemmatize preferred features
	relevant_words_lemmatized = [lemma.lemmatize(y) for y in relevant_words]

	# remove stopwords
	tokenized_policies = [[y for y in x if not y in stop_words] for x in tokenized_policies]
	
	# only included preferred features
	tokenized_policies = [[y for y in x if y in relevant_words_lemmatized] for x in tokenized_policies]
	
	print(relevant_words_lemmatized)
	print(tokenized_policies)
	print("length of policy: " + str(len(tokenized_policies)))
	print("\n")
	
	detokenized_policies = []
	for i in range(len(policies_list)):
		t = ' '.join(tokenized_policies[i])
		detokenized_policies.append(t)
	return detokenized_policies


if __name__ == '__main__':
	df = pd.read_csv("data/train.csv", encoding="ISO-8859-1")
	
	df_pp_read = pd.read_excel("data/PP_comparison_classification.xlsx")
	
	# SELECT ONLY URLS THAT ARE EITHER RELEVANT OR NOT RELEVANT WITH EXCLUSION CRITERIA CODE E1 (NOT RELEVANT)
	df_pp_relevant = df_pp_read[(df_pp_read['included'] == 'y') | (df_pp_read['included'] == 'e1')]
	
	df_pp_relevant_scoped = df_pp_relevant[
		['URL', 'UR_explicitly_mentioned', 'access', 'rectification', 'erasure', 'restriction',
		 'data_portability', 'object', 'automated_decision_making']]
	
	df_user_rights = df_pp_relevant[['access', 'rectification', 'erasure', 'restriction',
	                                 'data_portability', 'object', 'automated_decision_making']]
	
	# plot_stats(df_user_rights)
	
	# LIST OF URLs
	url_list = df_pp_relevant_scoped['URL'].tolist()
	
	# search for corresponding
	policy_list, policy_list_text = read_texts("data/privacy_policies", 140)
	
	url_list_policies = select_policies(url_list, policy_list, policy_list_text)
	url_list_policies = pre_processing(url_list_policies)
	
	# insert column with the corresponding privacy policies text
	df_pp_relevant_scoped.insert(2, 'policy', url_list_policies)
	df_pp_relevant.insert(2, 'policy', url_list_policies)
	
	# select df where policies are empty
	df_pp_relevant_scoped[(df_pp_relevant_scoped['policy'] == "")]
	pd.set_option("display.max_rows", None, "display.max_columns", None)  # print all values of dataframe
	
	# what is the policy distribution?
	df_test = df_pp_relevant[(df_pp_relevant['policy'] != "")]
	df_test[(df_test['included'] == 'y')]
	
	# filter out empty policies [147 policies in total: 109 relevant, 48 irrelevant]
	df_pp = df_pp_relevant_scoped[(df_pp_relevant_scoped['policy'] != "")]
	
	# fill NaN with 0
	df_pp.UR_explicitly_mentioned = pd.to_numeric(df_pp.UR_explicitly_mentioned, errors='coerce').fillna(0).astype(
		np.int64)
	df_pp.access = pd.to_numeric(df_pp.access, errors='coerce').fillna(0).astype(np.int64)
	df_pp.rectification = pd.to_numeric(df_pp.rectification, errors='coerce').fillna(0).astype(np.int64)
	df_pp.erasure = pd.to_numeric(df_pp.erasure, errors='coerce').fillna(0).astype(np.int64)
	df_pp.restriction = pd.to_numeric(df_pp.restriction, errors='coerce').fillna(0).astype(np.int64)
	df_pp.data_portability = pd.to_numeric(df_pp.data_portability, errors='coerce').fillna(0).astype(np.int64)
	df_pp.object = pd.to_numeric(df_pp.object, errors='coerce').fillna(0).astype(np.int64)
	df_pp.automated_decision_making = pd.to_numeric(df_pp.automated_decision_making, errors='coerce').fillna(0).astype(
		np.int64)
	# check if there are other entries than 0 or 1
	df_pp[(df_pp['UR_explicitly_mentioned'] != 0) & (df_pp['UR_explicitly_mentioned'] != 1)]
	
	# split data to train and test set
	categories = ['UR_explicitly_mentioned', 'access', 'rectification', 'erasure', 'restriction',
	              'data_portability', 'object', 'automated_decision_making']
	train, test = train_test_split(df_pp, random_state=42, test_size=0.33, shuffle=True)
	X_train = train.policy
	X_test = test.policy
	print(X_train.shape)
	print(X_test.shape)
	
	# Define a pipeline combining a text feature extractor with multi lable classifier
	NB_pipeline = Pipeline([
		('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 3))),
		('clf', OneVsRestClassifier(MultinomialNB(
			fit_prior=True, class_prior=None))),
	])
	for category in categories:
		print('... Processing user right: {}'.format(category))
		# train the model using X_dtm & y
		NB_pipeline.fit(X_train, train[category])
		# compute the testing accuracy
		prediction = NB_pipeline.predict(X_test)
		print(prediction)
		print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
	
	"""
	Accuracy is very high, probably due to an imbalanced dataset:
	UR_explicitly_mentioned 0.73 : always classified to 1
	access 0.73 : always classified to 1
	rectification 0.67 : always classified to 1
	erasure 0.71 : always classified to 1
	restriction 0.41 : always classified to 0
	- after reducing: varies and 0.47
	data_portability 0.47 : always classified to 0
	object 0.57 : varies
	- after reducing: 0.59
	automated_decision_making 0.90: always classified to 0
	
	probable cause: imbalanced dataset -> overfitting
	147 policies: 109 relevant, 48 irrelevant
	"""
