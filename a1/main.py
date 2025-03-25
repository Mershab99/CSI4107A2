import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
#nltk.download('stopwords')


import requests

# Fetch the custom stopwords list
def fetch_stopwords(url):
    response = requests.get(url)
    stopwords = set(response.text.split())
    return stopwords

# Load custom stopwords
stop_words_url = 'https://www.site.uottawa.ca/~diana/csi4107/StopWords'
stop_words = fetch_stopwords(stop_words_url)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens


stemmer = PorterStemmer()

# Load corpus
def load_corpus(filename):
    corpus = {}
    with open(filename, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_text = doc['title'] + " " + doc['text']  # Combine title and text
            corpus[doc['_id']] = preprocess(doc_text)
    return corpus

# Build BM25 index
def build_index(corpus):
    corpus_texts = list(corpus.values())
    bm25 = BM25Okapi(corpus_texts)
    return bm25, list(corpus.keys())

# Load queries
def load_queries(filename):
    queries = {}
    with open(filename, 'r') as f:
        for line in f:
            query = json.loads(line)
            if int(query['_id']) % 2 == 1:  # Use only test queries
                queries[query['_id']] = preprocess(query['text'])
    return queries

# Retrieve and rank documents
# Retrieve and rank documents
def retrieve_and_rank(bm25, doc_ids, queries):
    results = []
    ranked_data = {}
    for qid, query_tokens in queries.items():
        scores = bm25.get_scores(query_tokens)
        ranked_docs = np.argsort(scores)[::-1][:100]  # Top 100 docs
        ranked_data[qid] = [(doc_ids[idx], scores[idx]) for idx in ranked_docs[:10]]
        for rank, idx in enumerate(ranked_docs, start=1):
            results.append(f"{qid} Q0 {doc_ids[idx]} {rank} {scores[idx]:.4f} run_name")
    return results, ranked_data

# Write results to file
def write_results(results, filename="Results.txt"):
    with open(filename, 'w') as f:
        for line in results:
            f.write(line + "\n")

# Generate README details
def print_readme(vocab, sample_tokens, ranked_data):
    print("\n=== README DATA ===")
    print(f"Vocabulary Size: {len(vocab)}")
    print("Sample 100 Tokens:", list(vocab)[:100])
    print("\nFirst 10 Answers for First 2 Queries:")
    count = 0
    for qid, docs in ranked_data.items():
        print(f"Query {qid}:")
        for rank, (doc_id, score) in enumerate(docs, start=1):
            print(f"  Rank {rank}: Doc {doc_id}, Score {score:.4f}")
        count += 1
        if count == 2:
            break

# Main execution
corpus = load_corpus("scifact/corpus.jsonl")
vocab = set(word for doc in corpus.values() for word in doc)
bm25, doc_ids = build_index(corpus)
queries = load_queries("scifact/queries.jsonl")
results, ranked_data = retrieve_and_rank(bm25, doc_ids, queries)
write_results(results)
print_readme(vocab, list(vocab)[:100], ranked_data)
