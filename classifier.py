import csv
from typing import NamedTuple, List, Dict
from collections import Counter, defaultdict

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


class Document(NamedTuple):
    doc_id: int # NB data files index starting from 1
    sentence: List[str]
    classification_label: int

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  sentence: {self.sentence}\n" +
            f"  label: {self.classification_label}\n")

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



def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for word in doc.sentence():
            words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)

    for word in doc.sentence():
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




def get_documents(file):
    docs = []
    with open(file, "r", encoding="utf8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            docs.append(Document(int(line[0]), word_tokenize(line[2]), int(line[1])))
    return docs

def experiment():
    plant_docs = get_documents('./raw_data/plant.tsv')
    tank_docs = get_documents('./raw_data/tank.tsv')  
    perplace_docs = get_documents('./raw_data/perplace.tsv')   
    smsspam_docs = get_documents('./raw_data/smsspam.tsv')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
    ]

    return

if __name__ == '__main__':
    experiment()