import os
from googlesearch import search
from newspaper import Article
from six.moves.urllib.parse import urlparse
import nltk

'''
Author: Abdel-Jaouad Aberkane, Ghent University

- Scrape policies from Google
- Scrape list of URLs from Alexa
-

The script is based on the following sources:
#  https://pythondata.com/quick-tip-consuming-google-search-results-to-use-for-web-scraping/
'''

class WritePolicies:
	
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
	
	# Write parsed privacy policies to .TXT files
	def write2text(policies, dir):
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
				
				# print(pp_url)
				pp_file.writelines([pp_title, pp_url, pp_dop, pp_keywords, pp_full_text, pp_sep, "\n\n\n"])
				pp_file.close()
			except UnicodeEncodeError as ue:
				print(ue)