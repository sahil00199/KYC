import sys, os
from utils import separator, domainIndexToNameMapping

input_dir = "/mnt/blossom/data/sahilshah/sentiment/dataTf/Electronics::Video_Games/"
output_dir = "./data/"

input_files = ['ood_test.csv', 'dev.csv', 'train.csv', 'test.csv']
output_folders = ['ood', 'dev', 'train', 'test']

for input_filename, output_folder in zip(input_files, output_folders):
	file = open(os.path.join(input_dir, input_filename))
	lines = file.readlines()
	file.close()
	data = {}
	for line in lines:
		line = line.strip().split(separator)
		if len(line) == 0:
			continue
		else:
			assert len(line) == 3
			label = line[0]
			review = line[2]
			domain_id = int(line[1][:-1])
			if domain_id not in data.keys():
				data[domain_id] = []
			data[domain_id].append((label, review))
	for domain_id in data.keys():
		domain_name = domainIndexToNameMapping[domain_id]
		output_filename = os.path.join(os.path.join(output_dir, output_folder), domain_name + ".txt")
		output_file = open(output_filename, 'w')
		for label, review in data[domain_id]:
			output_file.write(label + "," + review)
			output_file.write("\n")
		output_file.close()
