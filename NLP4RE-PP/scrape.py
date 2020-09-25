import socket
import re
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from googlesearch import search
from newspaper import Article
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from simplified_scrapy.simplified_doc import SimplifiedDoc

'''
Author: Abdel-Jaouad Aberkane, Ghent University

- This class consists of methods that focus on scraping policies based on their URL.

Sources:
# https://pythondata.com/quick-tip-consuming-google-search-results-to-use-for-web-scraping/
# https://gist.github.com/graham-thomson/e9bf65ff17d214b144f91680cb81d438
'''


class Scrape:
	"""
	Goal: search policies from Google
	Args: search-query, number of desired policies
	Output: list of parsed policies as newspaper objects
	"""
	@staticmethod
	def scrape_policies_google(query, n_policies):
		policies = []
		for url in search(query, tld='com', lang='en', start=0, stop=n_policies):
			try:
				# print(url)
				article = Article(url)
				# Downloads the link’s HTML content
				article.download()
				article.parse()  # Parse the article
				article.nlp()
				policies.append(article)
			except:
				pass
		return policies
	
	# Adapted code from: https://gist.github.com/graham-thomson/e9bf65ff17d214b144f91680cb81d438
	# Alexa doesn't provide us anymore with categorization, choose top 50 in general instead
	@staticmethod
	def get_top_alexa_sites():
		# category = category.title()
		# categories = ['Adult', 'Arts', 'Business', 'Computers', 'Games', 'Health', 'Home',
		#               'Kids and Teens', 'News', 'Recreation', 'Reference', 'Regional', 'Science',
		#               'Shopping', 'Society', 'Sports', 'World']
		# assert (category in categories), "Category {} not in category list: {}".format(category, ", ".join(categories))
		# alexa_url = """http://www.alexa.com/topsites/category/Top/{}""".format(category)
		alexa_url = "http://www.alexa.com/topsites/"
		soup = BeautifulSoup(requests.get(alexa_url).content, "html.parser")
		divs = soup.find_all("div")
		
		top_sites = []
		
		for i in range(len(divs)):
			try:
				if divs[i]['class'] == [u'td', u'DescriptionCell']:
					div_links = divs[i].find_all('a')
					top_sites += div_links[0].contents
			except KeyError:
				continue
		
		def format_url(url):
			parsed_url = urlparse(url)
			if len(parsed_url.scheme) > 0 and len(parsed_url.netloc) > 0:
				return str(url.lower())
			elif len(parsed_url.scheme) == 0 and len(parsed_url.netloc) == 0 and len(parsed_url.path) > 0:
				return "http://{}".format(parsed_url.path.lower())
			else:
				return str(url.lower())
		
		return [format_url(ts_url) for ts_url in top_sites]
	
	# Extract corresponding privacy policies from a list of URLs
	@staticmethod
	def extract_policies_url_from_sites(websites):
		print("Extract policies ..")
		potential_policies = []
		policies = []
		for url in websites:
			print("URL under review: " + url + "\n")
			try:
				r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
				doc = SimplifiedDoc(r.content.decode('utf-8'))  # incoming HTML string
				lst = doc.listA(url=url)  # get all links
				
				for a in lst:
					if (("policy" in a['url']) or ("policies" in a['url']) or ("privacy" in a['url'])):
						# save all potential candidates
						potential_policies.append(a['url'])
				
				# take shortest string
				if len(potential_policies) > 0:
					policies.append(min(potential_policies, key=len))
					potential_policies = []
			
			except:
				pass
			
			print("Collected policies: ")
			print(policies)
			print("\n")
		return policies
	
	# Scrape html content from URLs
	@staticmethod
	def scrape_policies_url(policies):
		parsed_policies = []
		for url in policies:
			try:
				article = Article(url)
				article.download()  # Downloads the link’s HTML content
				article.parse()  # Parse the article
				article.nlp()
				# print(article.text)
				parsed_policies.append(article)
			# except article.ArticleException as ae:
			# print(ae)
			except:
				pass
		# avoid too many requests error from Google
		
		return parsed_policies
	
	@staticmethod
	def collect_policies(scraped_urls):
		policies = []
		counter = 0
		
		for url in scraped_urls:
			counter = counter + 1
			query = "site:\"" + url + "\" privacy policy"
			print("Query " + str(counter) + " " + query)
			policy = Scrape.scrape_policies(query)
			policies.append(policy)
		return policies
	
	# Extracts headers from parsed privacy policies and return dataframe
	@staticmethod
	def find_headers(policies):
		pre_dataframe = [[]]
		for policy in policies:
			
			req = Request(policy.url, headers={'User-Agent': 'Mozilla/5.0'})
			try:
				html = urlopen(req, timeout=1)
				policy_list = []
				
				# fill first element of the list (first cell of row) with the relevant website title
				policy_list.append(urlparse(policy.url).netloc + ".txt")
				# html = urlopen('https://automattic.com/privacy/')
				bs = BeautifulSoup(html, "html.parser")
				# 6 levels of HTML-headers
				titles = bs.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
				# print('List all the header tags :', *titles, sep='\n\n')
				
				for header in bs.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
					# print(header.text.strip())
					policy_list.append(header.text.strip())
				# print("*** *** *** END")
				pre_dataframe.append(policy_list)
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
		
		policies_dataframe = pd.DataFrame(pre_dataframe)
		return policies_dataframe
