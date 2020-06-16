from sklearn.feature_extraction.text import CountVectorizer
from utils import separator
from sklearn.feature_extraction import stop_words 
stop_words = list(stop_words.ENGLISH_STOP_WORDS)
import pickle
import os
import numpy as np

def is_out_of_distr(domainId):
        if domainId in ood_domains: return True
        return False

def is_in_distr(domainId):
        return not is_out_of_distr(domainId)

rawDataDir = 'data/'
folderName = 'data/processed/'

if not os.path.exists(folderName):
        os.mkdir(folderName)

train, test, ood_test, dev = [], [], [], []
domain_name_to_id_mapping = {}
documents = {}

# suffix is to differentiate between domains - t = train, d = dev, i = in distribution test, o = out of distribution test
print("Reading data...")
# label, domain, review
for suffix, folder_name, l in zip('tdio', ['train', 'dev', 'test', 'ood'], [train, dev, test, ood_test]):
    folder_name = os.path.join(rawDataDir, folder_name)
    filenames = [x for x in os.listdir(folder_name) if x[-4:] == ".txt"]
    for filename in filenames:
        domain_name = filename.replace(".txt", "")
        if domain_name not in domain_name_to_id_mapping:
            domain_name_to_id_mapping[domain_name] = len(domain_name_to_id_mapping)
        domain_id = str(domain_name_to_id_mapping[domain_name]) + suffix
        if domain_id not in documents.keys():
            documents[domain_id] = []
        file = open(os.path.join(folder_name, filename))
        lines = file.readlines()
        file.close()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            label = int(line[0])
            assert line[1] == ","
            review = line[2:]
            l.append((label, domain_id, review))
            documents[domain_id].append(review)

print("Done")





print("Writing csv files...")
## Write out the new files
for data, filename in zip([train, test, ood_test], ['train.txt', 'dev.txt', 'test.txt', 'ood_test.txt']):
        filename = os.path.join(folderName, filename)
        file = open(filename, 'w')
        file.write('\n'.join([separator.join([str(x) for x in line]) for line in data]))
        file.close()
print("Done")




## Data Sketch computation
fit_suffixes = ['t'] #alll domains with this suffix will be used to fit, and the rest will just be queries
vectorizer = CountVectorizer(stop_words = stop_words)
dist_vecs = {}

print ("Fitting Count vectorizer....")

corpus = [(k, v) for k, v in documents.items() if k[-1] in fit_suffixes]
domainNames = [k for k, v in corpus]
corpus = [" ".join(v) for k, v in corpus]
X = vectorizer.fit_transform(corpus)
X = X.toarray().astype(float)
word_counts = X.sum(axis = 0)
total_count = word_counts.sum()
p_w = word_counts / total_count
log_p_w = - np.log(p_w)
log_one_minus_p_w = - np.log(1.0 - p_w)

for i in range(len(domainNames)):
    N = X[i, :].sum()
    X[i, :] = log_p_w * X[i, :] + (N - X[i, :]) * log_one_minus_p_w
    X[i, :] /= np.linalg.norm(X[i, :])

dist_vecs = {}
for di, doc_item in enumerate(domainNames):
    dist_vecs[doc_item] = X[di, :]

queries = [(k, v) for k, v in documents.items() if k[-1] not in fit_suffixes]

queryDomainNames = [k for k, v in queries]
queries = [" ".join(v) for k, v in queries]
Xtest = vectorizer.transform(queries)
Xtest = Xtest.toarray().astype(float)

for i in range(len(queryDomainNames)):
    N = Xtest[i, :].sum()
    Xtest[i, :] = log_p_w * Xtest[i,:] + (N - Xtest[i, :]) * log_one_minus_p_w
    Xtest[i, :] /= np.linalg.norm(Xtest[i, :])

for di, doc_item in enumerate(queryDomainNames):
    dist_vecs[doc_item] = Xtest[di, :]

print ("Done!")

## Write the tf-idf
print("Writing distribution vectors")
pickle.dump(dist_vecs, open(os.path.join(folderName, "dist_vecs.pkl"), 'wb'))
print("Done")

