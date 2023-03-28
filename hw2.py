import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights):
    N = max(doc_freqs.values())
    freq = doc_freqs
    tf = compute_tf(doc, doc_freqs, weights)
    
    vec = defaultdict(float)

    # the calculation for IDF(t) was derived from Scikit-Learn which effectively handles edge cases
    for word in doc.author:
        vec[word] += tf[word] * (np.log2((N + 1) / (freq[word] + 1)) + 1)
    for word in doc.keyword:
        vec[word] += tf[word] * (np.log2((N + 1) / (freq[word] + 1)) + 1)
    for word in doc.title:
        vec[word] += tf[word] * (np.log2((N + 1) / (freq[word] + 1)) + 1)
    for word in doc.abstract:
        vec[word] += tf[word] * (np.log2((N + 1) / (freq[word] + 1)) + 1)

    return dict(vec)

def compute_boolean(doc, doc_freqs, weights):
    vec = defaultdict(bool)
    for word in doc.author:
        if doc_freqs[word] > 0:
            vec[word] == 1
        else:
            vec[word] == 0
    for word in doc.keyword:
        if doc_freqs[word] > 0:
            vec[word] == 1
        else:
            vec[word] == 0
    for word in doc.title:
        if doc_freqs[word] > 0:
            vec[word] == 1
        else:
            vec[word] == 0
    for word in doc.abstract:
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


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def get_recalls(results: List[int], relevant: List[int]):
    recalls = list()
    num_relevant = len(relevant)
    for k in range(0, len(relevant) + 1):
        recalls.append(k / num_relevant)
    return recalls

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents

    '''
    if recall == 0: # precision always 0 if recall is
        return 0 

    recalls = get_recalls(results, relevant)

    if recall in recalls:
        num_relevant = len(relevant)
        k = num_relevant * recall # number of correctly retrieved docs
        num_to_retrieve = 0
        counter = 0
        while counter < k:
            if results[num_to_retrieve] in relevant:
                counter += 1
            num_to_retrieve += 1

        B = k

        precision = B / num_to_retrieve
        return precision
    
    else: # interpolate case
        # determine two closest recalls:
        i = 0
        while recalls[i] < recall:
            i += 1
        
        return interpolate(recalls[i - 1], precision_at(recalls[i - 1], results, relevant), 
                           recalls[i], precision_at(recalls[i], results, relevant),
                           recall)

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    precision = 0
    for i in range(1, 11):
        precision += precision_at(i / 10, results, relevant)
    return precision / 10

def norm_recall(results, relevant):
    N = len(results)
    Rel = len(relevant)

    sum_of_ranks = 0
    for doc in relevant:
        sum_of_ranks += results[doc] + 1 # add 1 since ranks are counted from 1 not 0 and are thus offset by -1 
    sum_i = 0
    for i in range(1, Rel):
        sum_i += i

    return 1 - ((sum_of_ranks - sum_i)/(Rel*(N-Rel)))

def factorial(num):
    if num == 0:
        return 1
    return (num * factorial(num - 1))

def norm_precision(results, relevant):
    N = len(results)
    Rel = len(relevant)

    sum_of_ranks = 0
    sum_i = 0
    for doc in relevant:
        sum_of_ranks += np.log2(results[doc] + 1) # add 1 since ranks are counted from 1 not 0 and are thus offset by -1 
    for i in range(1, Rel):
        sum_i += np.log2(i)

    return 1 - ((sum_of_ranks - sum_i)  / (N * np.log2(N) - (N - Rel) * np.log2(N - Rel) - Rel * np.log2(Rel))) # approximate


### Extensions

# TODO: put any extensions here


### Search

def experiment():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

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
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            print(query_vec)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

        #return  # TODO: just for testing; remove this when printing the full table
    return


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()


if __name__ == '__main__':
    experiment()