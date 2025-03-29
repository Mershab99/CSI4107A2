"""
Assignment 2: Results File
Author: Buchra Omar (300174419)
"""
import pandas as pd
from collections import defaultdict

# Paths to input files
reranked_path = "Reranked_Results.txt"
bm25_path = "BM25_Results.txt"
qrels_path = "qrels/test.tsv"
results_output_path = "Results.txt"

# Load qrels
qrels_df = pd.read_csv(qrels_path, sep="\t", skiprows=1, names=["query_id", "doc_id", "relevance"])
qrels_df["query_id"] = qrels_df["query_id"].astype(int)
qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
qrels_df["relevance"] = qrels_df["relevance"].astype(int)


test_query_ids = set(qrels_df["query_id"].unique())

# Create relevance dictionary
qrels_dict = defaultdict(set)
for _, row in qrels_df.iterrows():
    if row["relevance"] > 0:
        qrels_dict[row["query_id"]].add(row["doc_id"])

# Function to load TREC format results
def load_trec_results(filepath):
    results = defaultdict(list)
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = int(parts[0])
                doc_id = parts[2]
                results[query_id].append(doc_id)
    return results

# Evaluation function
def evaluate(results, qrels_dict):
    map_scores = []
    p10_scores = []

    for query_id, retrieved_docs in results.items():
        relevant_docs = qrels_dict.get(query_id, set())
        if not relevant_docs:
            continue

        top_10 = retrieved_docs[:10]
        p10 = sum(1 for doc in top_10 if doc in relevant_docs) / 10.0
        p10_scores.append(p10)

        num_relevant = 0
        precisions = []
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precisions.append(num_relevant / rank)
        ap = sum(precisions) / len(relevant_docs) if relevant_docs else 0
        map_scores.append(ap)

    return round(sum(map_scores) / len(map_scores), 4), round(sum(p10_scores) / len(p10_scores), 4)

# Load system outputs
bm25_results = load_trec_results(bm25_path)
rerank_results = load_trec_results(reranked_path)

# Evaluate
bm25_map, bm25_p10 = evaluate(bm25_results, qrels_dict)
rerank_map, rerank_p10 = evaluate(rerank_results, qrels_dict)

# Output scores
print("BM25 MAP:", bm25_map, "P@10:", bm25_p10)
print("Neural Re-rank MAP:", rerank_map, "P@10:", rerank_p10)

# Filter reranked results for test queries
with open(reranked_path, "r") as infile:
    filtered_lines = [
        line for line in infile
        if int(line.split()[0]) in test_query_ids
    ]

# Write to Results.txt
with open(results_output_path, "w") as f:
    f.writelines(filtered_lines)

print("Filtered Results.txt created using best system (Neural Re-rank).")
