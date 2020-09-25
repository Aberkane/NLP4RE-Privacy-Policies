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


def scrape_google(scraper, writer, query, n_policies):
	policies = scraper.scrape_policies_google(query, n_policies)
	writer.write2texts(policies, ("data\\" + query))


def scrape_google_headers(scraper, query, n_policies):
	policies = scraper.scrape_policies_google(query, n_policies)
	policies_dataframe = scraper.find_headers(policies)
	policies_dataframe.to_excel("data\\policy_headers_test.xlsx")


def get_policies_excel_URLs():
	excel_urls = scraper.urls_from_excel("data\\PP_comparison_test.xlsx", "PPs vergelijken")
	# Extract policies'URL from list of URLs
	excel_policies = scraper.extract_policies_url_from_sites(excel_urls[90:95])
	excel_policies = scraper.scrape_policies_url(excel_policies)
	writer.write2texts(excel_policies, "data\\PPComparison_90+_test")


if __name__ == '__main__':
	scraper = scrape.Scrape()
	writer = write_policies.WritePolicies()
	reader = read_policies.ReadPolicies()
	modeler = LSA_policies.LSAPolicies()

# SCRAPE & SAVE POLICIES FROM GOOGLE
# scrape_google(scraper, writer, 'privacy policy', 50)

# SCRAPE AND PARSE HEADERS TO EXCEL
# scrape_google_headers(scraper, 'privacy policy', 10)

# READ URLS FROM EXCEL AND EXTRACT THEIR POLICIES FROM THE NET
# get_policies_excel_URLs()

# GET URLS FROM ALEXA (TOP 50) AND SCRAPE
# category = 'Business'
# top_50_all = scrape.get_top_alexa_sites(category)
# policies = scrape.extract_policies_url_from_sites(top_50_all)
# policies = scrape.scrape_policies(policies)
# write_policies.write2text(policies, "Alexa-top50allCategories")

# CONDUCT LSA OR LDA TOPIC MODELING ON DATASET
# dir = "privacy_policies"
# n_topics = 10
# # dir = "privacy_policies_headers"
# # print("read policies")
# policies, titles = read_txts(dir, 150)
#
# processed_policies = LSA_policies.pre_processing(policies)
#
# # print(policies[2])
# # TFIDF_vectorizer(processed_policies, n_topics)
# LSA_policies.count_TFIDF_vectorizer(processed_policies, n_topics, True)
