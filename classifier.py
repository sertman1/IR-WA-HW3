import csv
import re
import itertools
from typing import NamedTuple, List, Dict
from collections import Counter, defaultdict

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize



### File IO and Processing

class Document(NamedTuple):
    doc_id: int # NB data files index starting from 1
    sentence: List[str]
    classification_label: int

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  sentence: {self.sentence}\n" +
            f"  label: {self.classification_label}\n")

def get_documents(file):
    docs = []
    with open(file, "r", encoding="utf8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            docs.append(Document(int(line[0]), word_tokenize(line[2]), int(line[1])))
    return docs

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def stem_doc(doc: Document):
    stemmed_sentence = list()
    for word in doc.sentence:
        stemmed_sentence.append(stemmer.stem(word))

    return Document(doc.doc_id, stemmed_sentence, doc.classification_label)

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    pruned_sentence = list()
    for word in doc.sentence:
        if word not in stopwords:
            pruned_sentence.append(word)

    return Document(doc.doc_id, pruned_sentence, doc.classification_label)

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix
def get_dist_decay_weights(sentence):
    pos_ambigious = get_index_of_ambigious_word(sentence)
    weightings = []

    i = 0
    while i < len(sentence):
        dist = abs(pos_ambigious - i)
        if dist == 0: # prevent division by zero
            weightings.append(0)
        else:
            weightings.append(1 / dist)


        i += 1

    return weightings

def get_stepped_weights(sentence):
    pos_ambigious = get_index_of_ambigious_word(sentence)
    weightings = []

    i = 0
    while i < len(sentence):
        dist = abs(pos_ambigious - i)
        if dist == 0:
            weightings.append(0)
        elif dist == 1: # adjacent
            weightings.append(6)
        elif dist == 2 or dist == 3:
            weightings.append(3)
        else:
            weightings.append(1)
        
        i += 1

    return weightings

def get_ertman_weighting_weights(sentence):
    return 1

def get_index_of_ambigious_word(sentence):
    i = 0
    for word in sentence:
        if len(word) > 3 and word[0] == "." and word [1] == "X" and word[2] == "-":
            return i
        i += 1
    return

class TermWeights(NamedTuple):
    dist_decay: bool
    stepped: bool
    ertman: bool

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for word in doc.sentence:
            words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights):
    vec = defaultdict(float)
    computed_weights = []

    if weights.dist_decay == True:
        computed_weights = get_dist_decay_weights(doc.sentence)

    elif weights.stepped == True:
        computed_weights = get_stepped_weights(doc.sentence)

    else: # ertman weighting
        computed_weights = get_ertman_weighting_weights(doc.sentence)

    print(computed_weights)

    for word in doc.sentence:
        vec[word] += 1

    return dict(vec)

def compute_tfidf(doc, doc_freqs, weights):
    N = max(doc_freqs.values())
    freq = doc_freqs
    tf = compute_tf(doc, doc_freqs, weights)
    
    vec = defaultdict(float)

    # the calculation for IDF(t) was derived from Scikit-Learn which effectively handles edge cases
    for word in doc.sentence:
        vec[word] += tf[word] * (np.log2((N + 1) / (freq[word] + 1)) + 1)

    return dict(vec)

def compute_boolean(doc, doc_freqs, weights):
    vec = defaultdict(bool)

    for word in doc.sentence:
        if doc_freqs[word] > 0:
            vec[word] == 1
        else:
            vec[word] == 0

    return dict(vec)



### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

    # Suggestion: the computation of cosine similarity can be made more ecient by precomputing and
    # storing the sum of the squares of the term weights for each vector, as these are constants across vector
    # similarity comparisons.

def dice_sim(x, y):
    num = dictdot(x,y)
    if num == 0:
        return 0
    return (2 * num) / (norm(list(x.values())) * norm(list(y.values())))

def jaccard_sim(x, y):
    num = dictdot(x,y)
    if num == 0:
        return 0
    return num / ((norm(list(x.values())) * norm(list(y.values()))) - num)

def overlap_sim(x, y):
    num = dictdot(x,y)
    if num == 0:
        return 0
    return num / min(norm(list(x.values())),  norm(list(y.values())))



### Search

def experiment():
    data_sets = ['plant',
                 'tank', 
                 'perplace', 
                 'smsspam',
    ]

    term_funcs = {
        'tf': compute_tf,
        #'tfidf': compute_tfidf,
        #'boolean': compute_boolean,
    }

    sim_funcs = {
        'cosine': cosine_sim,
        #'jaccard': jaccard_sim,
        #'dice': dice_sim,
        #'overlap': overlap_sim
    }

    permutations = [
        data_sets,
        term_funcs,
        [False], #True],  # stem
        [False], #True],  # remove stopwords
        sim_funcs,
        [TermWeights(True, False, False),
         TermWeights(False, True, False),
         TermWeights(False, False, True)]
    ]

    for data_set, term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        # all_docs = get_documents('./raw_data/' + data_set + '.tsv')
        training_docs = get_documents('./training_data/' + data_set + '-train.tsv')
        dev_docs = get_documents('./dev_data/' + data_set + '-dev.tsv')

        training_docs = process_docs(training_docs, stem, removestop)
        dev_docs = process_docs(dev_docs, stem, removestop)

        training_term_freq = compute_doc_freqs(training_docs)
        dev_term_freq = compute_doc_freqs(dev_docs)
        training_vectors = [term_funcs[term](doc, training_term_freq, term_weights) for doc in training_docs]
        dev_vectors = [term_funcs[term](doc, dev_term_freq, term_weights) for doc in dev_docs]

        # compute centroid 
        v_profile1 = {}
        num_occurences1 = {}
        v_profile2 = {}
        num_occurences2 = {}
        i = 0
        for doc in training_docs:
            if doc.classification_label == 1:
                for key in training_vectors[i]:
                    if key in v_profile1:
                        v_profile1[key] += training_vectors[i][key]
                        num_occurences1[key] += 1
                    else:
                        v_profile1[key] = training_vectors[i][key]
                        num_occurences1[key] = 1
            else:
                for key in training_vectors[i]:
                    if key in v_profile2:
                        v_profile2[key] += training_vectors[i][key]
                        num_occurences2[key] += 1
                    else:
                        v_profile2[key] = training_vectors[i][key]
                        num_occurences2[key] = 1
            
            i += 1

        # normalize centroid
        for k in v_profile1:
            v_profile1[k] /= num_occurences1[k]
        for k in v_profile2:
            v_profile2[k] /= num_occurences2[k]

        # label dev data and keep track of percent correct
        total_correct = 0
        total_incorrect = 0
        i = 0
        for doc in dev_docs:
            sim1 = sim_funcs[sim](dev_vectors[i], v_profile1) # shift index to start at 0 
            sim2 = sim_funcs[sim](dev_vectors[i], v_profile2)
            if sim1 >= sim2: # model determined it is 1
                if doc.classification_label == 1:
                    total_correct += 1
                else:
                    total_incorrect += 1
            else: # model determined it is 2
                if doc.classification_label == 2:
                    total_correct += 1
                else:
                    total_incorrect += 1
            i += 1

        print(total_correct / (total_correct + total_incorrect))

    return

def process_docs(docs, stem, removestop):
    processed_docs = docs
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
    if stem:
        processed_docs = stem_docs(processed_docs)
    return processed_docs

if __name__ == '__main__':
    experiment()