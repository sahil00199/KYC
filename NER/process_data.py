import os
import tqdm
import codecs
import numpy as np
import pickle
from transformers import BertTokenizer

from sklearn.feature_extraction.text import CountVectorizer

"""
Helper script to process data written in a complying CONLL format and create the required domain signatures
"""

input_dir = "./data"
output_dir = "./data/processed"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

out_files = ["test.txt", "ood-test.txt", "train.txt"]
input_splits = ['test', 'ood', 'train']


def parse_raw_input():
    for input_split, out_file in zip(input_splits, out_files):
        output_file = open(os.path.join(output_dir, out_file), 'w')
        current_input_dir = os.path.join(input_dir, input_split)
        for filename in os.listdir(current_input_dir):
            domain_name = filename.replace(".txt", "") + "_" + input_split
            input_file = open(os.path.join(current_input_dir, filename))
            lines = input_file.readlines()
            input_file.close()
            for line in lines:
                line = line.strip().split()
                if len(line) == 0:
                    output_file.write("\n")
                elif len(line) > 0:
                    assert len(line) == 2
                    output_file.write(" ".join(line + [domain_name]))
                    output_file.write("\n")
        output_file.close()

def preprocess():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    max_len = 128
    for out_fname in out_files:
        subword_len_counter = 0
        dataset = os.path.join(output_dir, out_fname)
        outfile = open('temp.txt', 'w')
        with open(dataset, "rt") as f_p:
            for line in f_p:
                if line[:6] == '#begin' or line[:4] == '#end': continue
                line = line.rstrip()

                if not line:
                    outfile.write(line + '\n')
                    subword_len_counter = 0
                    continue

                token = line.split()[0]

                current_subwords_len = len(tokenizer.tokenize(token))

                # Token contains strange control characters like \x96 or \x95
                # Just filter out the complete line
                if current_subwords_len == 0:
                    continue

                if (subword_len_counter + current_subwords_len) > max_len:
                    outfile.write("\n")
                    outfile.write(line + '\n')
                    subword_len_counter = 0
                    continue

                subword_len_counter += current_subwords_len

                outfile.write(line + '\n')
        os.system('mv temp.txt ' + dataset)

def write_dist_vecs():
    # Write distributional vectors for each split type -- simulating the server and client side behaviour
    documents = {}
    for oi, out_fname in enumerate(out_files[:1]):
        with codecs.open(os.path.join(output_dir, out_fname), 'r', 'utf-8') as f:
            count = -1
            for l in f:
                count += 1
                l = l.strip()
                if l.startswith('#') or len(l) == 0:
                    continue

                token, tag, dom = l.split()
                if dom not in documents.keys(): documents[dom] = []
                documents[dom].append(token)

    for domain_name in documents.keys():
        documents[domain_name] = ' '.join(documents[domain_name])

    # collect distributional vector for each domain 
    documents = documents.items()
    corpus = [doc[1] for doc in documents]
    
    from sklearn.feature_extraction import stop_words 
    stop_words = list(stop_words.ENGLISH_STOP_WORDS)
    
    vectorizer = CountVectorizer(stop_words = stop_words)
    X = vectorizer.fit_transform(corpus)

    X = X.toarray().astype(float)
    word_counts = X.sum(axis = 0)
    total_count = word_counts.sum()
    p_w = word_counts / total_count
    log_p_w = - np.log(p_w)
    log_one_minus_p_w = - np.log(1.0 - p_w)

    for i in range(len(documents)):
        N = X[i, :].sum()
        X[i, :] = log_p_w * X[i, :] + (N - X[i, :]) * log_one_minus_p_w
        X[i, :] /= np.linalg.norm(X[i, :])

    dist_vecs = {}
    for di, doc_item in enumerate(documents):
        dist_vecs[doc_item[0]] = X[di, :]
    ############################################## Now query using ood-test set ##############################################
    testDocuments = {}
    for oi, out_fname in enumerate(out_files[1:]):
        with codecs.open(os.path.join(output_dir, out_fname), 'r', 'utf-8') as f:
            for l in f:
                l = l.strip()
                if l.startswith('#') or len(l) == 0:
                    continue
                token, tag, dom = l.split()
                if dom not in testDocuments.keys(): testDocuments[dom] = []
                testDocuments[dom].append(token)

    for domain_name in testDocuments.keys():
        testDocuments[domain_name] = ' '.join(testDocuments[domain_name])

    testDocuments = testDocuments.items()
    testCorpus = [x for _, x in testDocuments]
    Xtest = vectorizer.transform(testCorpus)
    Xtest = Xtest.toarray().astype(float)

    for i in range(len(testDocuments)):
        N = Xtest[i, :].sum()
        Xtest[i, :] = log_p_w * Xtest[i,:] + (N - Xtest[i, :]) * log_one_minus_p_w
        Xtest[i, :] /= np.linalg.norm(Xtest[i, :])

    for di, doc_item in enumerate(testDocuments):
        dist_vecs[doc_item[0]] = Xtest[di, :]
    ###########################################################################################################################
    

    with open(os.path.join(output_dir, "dist_vecs.pkl"), "wb") as f:
       pickle.dump(dist_vecs, f)

    dist_items = [_ for _ in dist_vecs.items()]
        
if __name__ == '__main__':
    print("Reading inputs...")
    parse_raw_input()
    print("Done")
    print("Truncating long sentences and performing other preprocessing...")
    preprocess()
    print("Preprocessing Done!")
    print("Computing domain sketch...")
    write_dist_vecs()
    print("Done")
