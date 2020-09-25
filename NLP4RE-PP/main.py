import scrape
import write_policies
import read_policies
import LSA_policies

'''
Author: Abdel-Jaouad Aberkane, Ghent University

In this file we do the following:
- Scrape policies
- Read policies or URLs
- Write policies to txt of xlrs
- Topic modeling on scraped policies
'''

dir_headers = "data\\policy_headers_2.xlsx"
dir_policies = "data\\privacy_policies"


def scrape_google(query, n_policies):
	policies = scraper.scrape_policies_google(query, n_policies)
	writer.write2texts(policies, ("data\\" + query))


def scrape_google_headers(query, n_policies):
	policies = scraper.scrape_policies_google(query, n_policies)
	policies_dataframe = scraper.find_headers(policies)
	policies_dataframe.to_excel("data\\policy_headers_2.xlsx")


def scrape_alexa():
	top_50_all = scraper.get_top_alexa_sites()
	policies = scraper.extract_policies_url_from_sites(top_50_all)
	policies = scraper.scrape_policies_url(policies)
	writer.write2texts(policies, "data\\Alexa-top50allCategories_test")


def get_policies_excel_URLs():
	excel_urls = reader.urls_from_excel("data\\PP_comparison_test.xlsx", "PPs vergelijken")
	# Extract policies'URL from list of URLs
	excel_policies = scraper.extract_policies_url_from_sites(excel_urls[90:95])
	excel_policies = scraper.scrape_policies_url(excel_policies)
	writer.write2texts(excel_policies, "data\\PPComparison_90+_test")


def topic_modeling(n_topics, min_policy_len, headers=False):
	if not headers:
		print("Topic modeling on textual bodies...")
		policies, titles = reader.read_texts(dir_policies, min_policy_len)
		processed_policies = modeler.pre_processing(policies)
		modeler.count_TFIDF_vectorizer(processed_policies, n_topics, True)
	else:
		print("Topic modeling on headers...")
		header_list = reader.read_headers(dir_headers, min_policy_len)
		processed_policies = modeler.pre_processing(header_list)
		modeler.count_TFIDF_vectorizer(processed_policies, n_topics, True)


if __name__ == '__main__':
	scraper = scrape.Scrape()
	writer = write_policies.WritePolicies()
	reader = read_policies.ReadPolicies()
	modeler = LSA_policies.LSAPolicies()
	n_words_policy = 150
	n_topics = 10
	
	# SCRAPE & SAVE POLICIES FROM GOOGLE
	# scrape_google('privacy policy', 5)
	
	# SCRAPE AND PARSE HEADERS TO EXCEL
	# scrape_google_headers('privacy policy', 500)
	
	# READ URLS FROM EXCEL AND EXTRACT THEIR POLICIES FROM THE NET
	# get_policies_excel_URLs()
	
	# GET URLS FROM ALEXA (TOP 50) AND SCRAPE
	# scrape_alexa()
	
	# CONDUCT LSA TOPIC MODELING ON DATASET OF PRIVACY POLICIES
	# topic_modeling(n_topics, n_words_policy)
	
	# CONDUCT LSA TOPIC MODELING ON PRIVACY POLICY HEADERS
	topic_modeling(n_topics, n_words_policy, True)
