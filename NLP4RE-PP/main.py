import scrape
import write_policies
import read_policies
import LSA_policies

'''
Author: Abdel-Jaouad Aberkane, Ghent University

- Scrape policies
- Read policies or URL
- Write policies to txt of xlrs
- Topic modeling on scraped policies
'''

if __name__ == '__main__':
	pass
	# SCRAPE POLICIES FROM GOOGLE
	# query = 'privacy policy'
	# policies = scrape.scrape_policies_google(query, 10)
	# write_policies.write2txts(policies, query)
	
	# SCRAPE AND PARSE HEADERS
	# query = 'privacy policy'
	# policies = scrape_policies(query, 100)
	# policies_dataframe = scrape.find_headers(policies)
	# policies_dataframe.to_excel("policy_headers.xlsx")
	# print("finished")
	
	# READ URLS FROM EXCEL AND EXTRACT THEIR POLICIES FROM THE NET
	# print("Extract URLs from Excel-file")
	# excel_urls = scrape.urls_from_excel("PP_comparison.xlsx", "PPs vergelijken")
	#
	# # PPs voor index 90 reeds gedaan
	# print(excel_urls[90:])
	#
	# print("Extract policies from URLs ...")
	# policies = scrape.extract_policies_url_from_sites(excel_urls[90:])
	# print("Scrape policies ...")
	# policies = scrape_policies(policies)
	# print("Write to text")
	# write_policies.write2text(policies, "PPComparison90+")

	# GET URLS FROM ALEXA (TOP 50) AND SCRAPE
	# category = 'Business'
	# top_50_all = scrape.get_top_alexa_sites(category)
	# policies = scrape.extract_policies_url_from_sites(top_50_all)
	# policies = scrape.scrape_policies(policies)
	# write_policies.write2text(policies, "Alexa-top50allCategories")
	
	# CONDUCT LSA OR LDA TOPIC MODELING ON DATASET
	dir = "privacy_policies"
	n_topics = 10
	# dir = "privacy_policies_headers"
	# print("read policies")
	policies, titles = read_txts(dir, 150)
	
	processed_policies = LSA_policies.pre_processing(policies)
	
	# print(policies[2])
	# TFIDF_vectorizer(processed_policies, n_topics)
	LSA_policies.count_TFIDF_vectorizer(processed_policies, n_topics, True)