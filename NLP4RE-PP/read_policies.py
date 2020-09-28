import re
import pandas as pd
from langdetect import detect
import os.path

'''
Author: Abdel-Jaouad Aberkane, Ghent University

- This class consists of methods that focus reading policies or URLs from files.
'''


class ReadPolicies:
	
	"""
	Goal: extract URLs from Excel file and clean them
	Args: directory of the Excel file, relevant excel sheet
	Output: list URLs
	"""
	@staticmethod
	def urls_from_excel(doc, sheet):
		print("Extracting URLs from XLSX-file")
		PPcomp = pd.read_excel(doc, sheet_name=sheet)
		PPlist = PPcomp.iloc[:, 0].to_list()
		cleanPP = [re.sub('[,!?&“”():"]', '', str(x)) for x in PPlist if x is not None]
		cleanPP = [re.sub('[" "]', '', str(x)) for x in cleanPP if x is not None]
		cleanPP = [re.sub('[0-9]', '', str(x)) for x in cleanPP if x is not None]
		
		for i, s in enumerate(cleanPP):
			if not ('http' in s):
				cleanPP[i] = "https://" + s
		
		return cleanPP
	
	"""
	Simple function to cast list to string
	# https://www.geeksforgeeks.org/python-program-to-convert-a-list-to-string/
	"""
	@staticmethod
	def list_to_string(s):
		# initialize an empty string
		str1 = ""
		
		# traverse in the string
		for ele in s:
			str1 += ele
		
		# return string
		return str1
	
	"""
	Goal: reads privacy policies from the data folder and parses them into a list
	Args: directory of the policies, minimal number of words of policy (e.g. to filter out irrelevant texts)
	Output: list of policies
	"""
	@staticmethod
	def read_texts(dir, n_words_policy):
		print("Extracting texts from:" + dir)
		all_files = os.listdir(dir)
		policies_list = []
		titles = []
		
		# filter on txt files (redundant, but safety first)
		txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
		
		for file in txt_files:
			# print(file)
			with open((dir + "\\" + file), "r", encoding="utf8") as policy:
				data = policy.readlines()
				policy_text = ReadPolicies.list_to_string(data[5:])
				
				if len(policy_text.split()) > n_words_policy:
					lang = detect(policy_text)
					if lang == 'en':
						# print("het is engels")
						# print(lang)
						policies_list.append(policy_text)
						title = (data[0].rstrip("\n").replace("Title: ", ""))
						titles.append(title)
					else:
						# print("het is geen engels")
						# print(lang)
						pass
		
		# print(title)
		# print(policy.readlines())
		
		# with open(os.path.join(path, file_name), "r") as fin:
		# 	for line in fin.readlines():
		# 		text = line.strip()
		# 		documents_list.append(text)
		# print("Total Number of Documents:", len(documents_list))
		# titles.append(text[0:min(len(text), 100)])
		return policies_list, titles
	
	"""
	Goal: read rows from Excel file and parse it to a string
	Args: directory of the Excel file
	Output: list of parsed header policies
	"""
	@staticmethod
	def read_headers(dir_headers, n_words_policy):
		print("Extracting headers from XLSX-file ...")
		# dfread = pd.read_excel(r'data\policy_headers.xlsx')
		dfread = pd.read_excel(dir_headers)
		
		# Create an empty list
		header_list = []
		
		# Iterate over each row
		for index, rows in dfread.iterrows():
			row_tmp = ""
			for element in rows:
				if len(str(element)) > 3 and "txt" not in element:  # exclude urls and NAN
					row_tmp = row_tmp + " " + str(element)
			
			if len(row_tmp) > n_words_policy:
				lang = detect(row_tmp)
				if lang == 'en':
					header_list.append(row_tmp)
		
		return header_list
