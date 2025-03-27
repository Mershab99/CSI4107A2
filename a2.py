import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import requests
from sentence_transformers import SentenceTransformer, util
import tensorflow_hub as hub

# === NLTK Downloads ===
nltk.download('punkt')
nltk.download('punkt_tab')


# === Fetch Custom Stopwords List ===
def fetch_stopwords(url):
    response = requests.get(url)
    stopwords = set(response.text.split())
    return stopwords


# Load custom stopwords
stop_words_url = 'https://www.site.uottawa.ca/~diana/csi4107/StopWords'
stop_words = fetch_stopwords(stop_words_url)

# === Preprocessing Function ===
stemmer = PorterStemmer()


def preprocess(text):
    """Preprocesses text: lowercasing, punctuation removal, tokenization, stopword removal, and stemming"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens


# === Load Corpus ===
def load_corpus(filename):
    """Loads corpus from JSONL file and preprocesses documents"""
    corpus = {}
    with open(filename, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_text = doc['title'] + " " + doc['text']
            corpus[doc['_id']] = preprocess(doc_text)
    return corpus


# === Build BM25 Index ===
def build_index(corpus):
    """Builds BM25 index from preprocessed corpus"""
    corpus_texts = list(corpus.values())
    bm25 = BM25Okapi(corpus_texts)
    return bm25, list(corpus.keys())


# === Load Queries ===
def load_queries(filename):
    """Loads test queries and preprocesses them"""
    queries = {}
    with open(filename, 'r') as f:
        for line in f:
            query = json.loads(line)
            if int(query['_id']) % 2 == 1:  # Use only test queries
                queries[query['_id']] = preprocess(query['text'])
    return queries


# === Retrieve and Rank Documents with BM25 ===
def retrieve_and_rank(bm25, doc_ids, queries):
    """Retrieves and ranks documents using BM25, returns top 100 results"""
    results = []
    ranked_data = {}

    for qid, query_tokens in queries.items():
        scores = bm25.get_scores(query_tokens)
        ranked_docs = np.argsort(scores)[::-1][:100]  # Top 100 docs
        ranked_data[qid] = [(doc_ids[idx], scores[idx]) for idx in ranked_docs[:10]]

        for rank, idx in enumerate(ranked_docs, start=1):
            results.append(f"{qid} Q0 {doc_ids[idx]} {rank} {scores[idx]:.4f} bm25")

    return results, ranked_data


# === Neural Re-ranking with BERT and USE ===
# Initialize Models
model_bert = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence BERT
model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def generate_bert_embeddings(texts):
    """Generates sentence embeddings using BERT"""
    return model_bert.encode(texts, convert_to_tensor=True)


def generate_use_embeddings(texts):
    """Generates sentence embeddings using Universal Sentence Encoder (USE)"""
    return model_use(texts).numpy()


def rerank_with_embeddings(queries, bm25_results, corpus):
    """Re-ranks BM25 top 100 results using neural embeddings"""
    reranked_results = []

    for qid, results in bm25_results.items():
        doc_ids = [doc_id for doc_id, _ in results]
        docs = [" ".join(corpus[doc_id]) for doc_id in doc_ids]

        # Generate query embeddings
        query_text = " ".join(queries[qid])

        q_bert = generate_bert_embeddings([query_text])
        q_use = generate_use_embeddings([query_text])

        # Generate document embeddings
        d_bert = generate_bert_embeddings(docs)
        d_use = generate_use_embeddings(docs)

        # Compute similarity scores
        bert_scores = util.pytorch_cos_sim(q_bert, d_bert).flatten().tolist()
        use_scores = np.inner(q_use, d_use).flatten().tolist()

        # Combine similarity scores (average)
        combined_scores = [(doc_id, (bert + use) / 2)
                           for (doc_id, bert, use) in zip(doc_ids, bert_scores, use_scores)]

        # Sort by combined similarity score
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Store re-ranked results in TREC format
        for rank, (doc_id, score) in enumerate(combined_scores, start=1):
            reranked_results.append(f"{qid} Q0 {doc_id} {rank} {score:.4f} neural_rerank")

    return reranked_results


# === Output Results to File ===
def write_results(results, filename):
    """Writes results to file in TREC format"""
    with open(filename, 'w') as f:
        for line in results:
            f.write(line + "\n")


# === Display README Details ===
def print_readme(vocab, sample_tokens, ranked_data):
    """Prints README details with vocabulary size, sample tokens, and ranked data"""
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


# === Main Execution ===
if __name__ == "__main__":
    # Load corpus and build BM25 index
    corpus = load_corpus("scifact/corpus.jsonl")
    vocab = set(word for doc in corpus.values() for word in doc)
    bm25, doc_ids = build_index(corpus)

    # Load queries
    queries = load_queries("scifact/queries.jsonl")

    # BM25 retrieval
    bm25_results, top100_results = retrieve_and_rank(bm25, doc_ids, queries)

    # Write initial BM25 results
    write_results(bm25_results, "BM25_Results.txt")

    # Re-rank with neural embeddings
    reranked_results = rerank_with_embeddings(queries, top100_results, corpus)

    # Write re-ranked results
    write_results(reranked_results, "Reranked_Results.txt")

    # Display README data
    print_readme(vocab, list(vocab)[:100], top100_results)

    print("\n✅ Results saved successfully!")
    print("- BM25 results: BM25_Results.txt")
    print("- Neural re-ranked results: Reranked_Results.txt")
