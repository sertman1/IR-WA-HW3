import csv
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

def get_LRtokens(line): # For special adjacent tokens
    sentence = word_tokenize(line)
    ambigious_pos = get_index_of_ambigious_word(sentence)
    if ambigious_pos == -1: # SMSspam
        return sentence
    else:
        if ambigious_pos > 0:
            sentence[ambigious_pos - 1] = "L-" + sentence[ambigious_pos - 1]
        if ambigious_pos + 1 < len(line):
            sentence[ambigious_pos + 1] = "R-" + sentence[ambigious_pos + 1]

    return sentence

def get_documents(file, collocation):
    docs = []
    with open(file, "r", encoding="utf8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            if collocation == 1:
                docs.append(Document(int(line[0]), word_tokenize(line[2]), int(line[1])))
            else: # adjacent_seperate_LR
                docs.append(Document(int(line[0]), get_LRtokens(line[2]), int(line[1])))
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

def get_uniform_weights(sentence):
    weightings = [1] * len(sentence)
    return weightings

def get_index_of_ambigious_word(sentence):
    i = 0
    for word in sentence:
        if len(word) > 3 and word[0] == "." and word [1] == "X" and word[2] == "-":
            return i
        i += 1
    return -1

class TermWeights(NamedTuple):
    uniform: bool
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
    computed_weights = get_uniform_weights(doc.sentence)

    i = 0
    while i < len(doc.sentence):
        term = doc.sentence[i]

        if len(term) > 3 and term[0] == "." and term [1] == "X" and term[2] == "-":
            i += 1
        else:
            vec[term] += computed_weights[i] 
            i += 1

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
    search(False, True, TermWeights(True, False, False, False), 1, 'cosine')
    return

def search(stem, removestop, term_weights, collocation, sim):
    data_sets = [
        'tank',
        'plant', 
        'perplace', 
        'smsspam',
    ]

    term_funcs = {
        'tf': compute_tf,
    }

    sim_funcs = {
        'cosine': cosine_sim,
        #'jaccard': jaccard_sim,
        #'dice': dice_sim,
        #'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        data_sets
    ]

    results = {}
    for term, data_set in itertools.product(*permutations):
        
        training_docs = get_documents('./training_data/' + data_set + '-train.tsv', collocation)
        dev_docs = get_documents('./dev_data/' + data_set + '-dev.tsv', collocation)

        training_docs = process_docs(training_docs, stem, removestop)
        dev_docs = process_docs(dev_docs, stem, removestop)

        sense1_train_docs = get_sense1_docs(training_docs)
        sense2_train_docs = get_sense2_docs(training_docs)

        # create sense1 and sense2 training vectors
        sense1_train_term_freq = compute_doc_freqs(sense1_train_docs)
        sense2_train_term_freq = compute_doc_freqs(sense2_train_docs)
        sense1_train_vectors = [term_funcs[term](doc, sense1_train_term_freq, term_weights) for doc in sense1_train_docs]
        sense2_train_vectors = [term_funcs[term](doc, sense2_train_term_freq, term_weights) for doc in sense2_train_docs]

        # create development vectors
        dev_term_freq = compute_doc_freqs(dev_docs)
        dev_vectors = [term_funcs[term](doc, dev_term_freq, term_weights) for doc in dev_docs]


        V_sum1 = {}
        for vec in sense1_train_vectors:
            for term in vec:
                if term in V_sum1:
                    V_sum1[term] += vec[term]
                else:
                    V_sum1[term] = vec[term]
        

        V_sum2 = {}
        for vec in sense2_train_vectors:
            for term in vec:
                if term in V_sum2:
                    V_sum2[term] += vec[term]
                else:
                    V_sum2[term] = vec[term]

        logLikelihood = {}
        EPSILON = 0.2
        for term in V_sum2:
            if (term in V_sum1 and V_sum1[term] > 0):
                logLikelihood[term] = np.log( V_sum1[term] / V_sum2[term] )
            else:
                logLikelihood[term] = np.log( EPSILON / V_sum2[term])
        
        for term in V_sum1:
            if term not in V_sum2:
                logLikelihood[term] = np.log ( V_sum1[term] / EPSILON)

        i = 0
        sumofLL = 0
        for doc in dev_docs:
            vec = dev_vectors[i]
            for term in vec:
                if term in logLikelihood:
                    sumofLL += logLikelihood[term] * vec[term]

                if sumofLL > 0: 
                    predicted_class = 1
                elif sumofLL < 0:
                    predicted_class = 2
                else: # both classes equally likely
                    predicted_class = 0

            print("Vec #" + str(doc.doc_id + 1) + 
                  "\tTrue Class: " + str(doc.classification_label) + 
                  "\tPredicted Class: " + str(predicted_class) + 
                  "\tSum of LL: " + str(sumofLL))

            i += 1


    print(results)
    return

def process_docs(docs, stem, removestop):
    processed_docs = docs
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
    if stem:
        processed_docs = stem_docs(processed_docs)
    return processed_docs

# splits the documents into two separate lists of documents wheter they are sense 1 or 2
def get_sense1_docs(docs):
    sense1_docs = list()
    for doc in docs:
        if doc.classification_label == 1:
            sense1_docs.append(doc)
    return sense1_docs
def get_sense2_docs(docs):
    sense2_docs = list()
    for doc in docs:
        if doc.classification_label == 2:
            sense2_docs.append(doc)
    return sense2_docs

if __name__ == '__main__':
    experiment()