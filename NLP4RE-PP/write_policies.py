import os
from googlesearch import search
from newspaper import Article
from six.moves.urllib.parse import urlparse
import nltk

'''
Author: Abdel-Jaouad Aberkane, Ghent University

- This class focuses on writing extracted policies to text or excel files.
'''

class WritePolicies:
	
	@staticmethod
	def write2txt(policies):
		pp_file = open("privacy_policies.txt", "a", encoding='utf-8')
		separator = ', '
		
		for policy in policies:
			try:
				pp_title = "Title: " + policy.title + "\n"
				pp_url = "URL: " + policy.url + "\n"
				pp_dop = "Date of publishing: " + str(policy.publish_date) + "\n"
				pp_keywords = "Keywords: " + separator.join(policy.keywords) + "\n\n"
				pp_summary = "Summary: " + policy.summary + "\n\n"
				pp_full_text = "Policy: " + policy.text + "\n"
				pp_sep = "***************************************************************"
				
				pp_file.writelines([pp_title, pp_url, pp_dop, pp_keywords, pp_full_text, pp_sep, "\n\n\n"])
			except UnicodeEncodeError as ue:
				print(ue)
		pp_file.close()
	
	@staticmethod
	def write2text(policies, dir):
		separator = ', '
		
		if not os.path.exists(dir):
			os.makedirs("data\\" + dir)
		
		for policy in policies:
			try:
				pp_file_name = urlparse(policy.url).netloc + ".txt"
				pp_file = open(dir + "\\" + pp_file_name, "a", encoding='utf-8')
				
				pp_title = "Title: " + policy.title + "\n"
				pp_url = "URL: " + policy.url + "\n"
				pp_dop = "Date of publishing: " + str(policy.publish_date) + "\n"
				pp_keywords = "Keywords: " + separator.join(policy.keywords) + "\n\n"
				pp_summary = "Summary: " + policy.summary + "\n\n"
				pp_full_text = "Policy: " + policy.text + "\n"
				pp_sep = "***************************************************************"
				
				# print(pp_url)
				pp_file.writelines([pp_title, pp_url, pp_dop, pp_keywords, pp_full_text, pp_sep, "\n\n\n"])
				pp_file.close()
			except UnicodeEncodeError as ue:
				print(ue)

	"""
	Goal: write parsed privacy policies to TXTs
	Args: list of privacy policies as newspaper objects and desired save directory
	"""
	
	@staticmethod
	def write2texts(policies, dir):
		separator = ', '
	
		if not os.path.exists(dir):
			os.makedirs(dir)
	
		for policy in policies:
			try:
				pp_file_name = urlparse(policy.url).netloc + ".txt"
				pp_file = open(dir + "\\" + pp_file_name, "a", encoding='utf-8')
	
				pp_title = "Title: " + policy.title + "\n"
				pp_url = "URL: " + policy.url + "\n"
				pp_dop = "Date of publishing: " + str(policy.publish_date) + "\n"
				pp_keywords = "Keywords: " + separator.join(policy.keywords) + "\n\n"
				pp_summary = "Summary: " + policy.summary + "\n\n"
				pp_full_text = "Policy: " + policy.text + "\n"
				pp_sep = "***************************************************************"
	
				print(pp_url)
				pp_file.writelines([pp_title, pp_url, pp_dop, pp_keywords, pp_full_text, pp_sep, "\n\n\n"])
				pp_file.close()
			except UnicodeEncodeError as ue:
				print(ue)