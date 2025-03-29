
# CSI4107 Assignment 2

### Members
- Mershab Issadien (300027272):
  - Preprocessing corpus by lowercasing, removing punctuation, applying tokenization, removing stopwords and stemming.
  - BM25 IR retrieval system.
      - Building BM25 index and ranking documents based on similarity scores.
  - Neural Re-ranking using `Sentence-BERT` (`all-MiniLM-L6-v2`) and `Universal Sentence Encoder` (USE).
    - Reranking the BM25 results.
    - Generating embeddings for both queries and top 100 BM25 retrieved documents.
    - Calculating similarity scores using cosine similarity for BERT and inner product similarity for USE.
    - Combining the scores from both models and averaging them to get the final re-ranked results.
    - Writing the results in TREC format.

- Buchra Omar (300174419):
 - Created the `Results.txt` file from reranked output of BERT and USE scores.
 - Obtained the relevant results in TREC format while only including the test queries.
 - Completed full evaluation of BM25 and Neural Re-ranking systems on `test.tsv`.
 - Measured and reported final MAP and P@10 scores.
 - Created script `generate_results_and_evaluate.py` for reproducibilty and automated submission.
 


### **Functionality of the Program**

This program implements an **Information Retrieval (IR) system** using the **BM25 ranking model**. It processes a collection of scientific documents (from `corpus.jsonl`), indexes them, and ranks them against a set of test queries (from `queries.jsonl`). The results are saved in the required TREC format (`Results.txt`). The key components of the system are **preprocessing, indexing, retrieval, and evaluation**.

---

## **Functionality Overview**
### 1. **Preprocessing**
- Fetches a custom list of **stopwords** from the provided URL.
- Preprocesses the text by:
  - Lowercasing all words.
  - Removing punctuation.
  - Tokenizing the text into individual words.
  - Removing stopwords.
  - Applying **Porter stemming**.

### 2. **BM25 Retrieval**
- Builds the **BM25 index** from the preprocessed corpus.
- Ranks documents for each query based on BM25 similarity scores.
- Retrieves the **top 100 documents** per query.
- Writes the BM25 results in **TREC format** to `BM25_Results.txt`.

### 3. **Neural Re-ranking**
- For each query:
  - Retrieves the **top 100 BM25 documents**.
  - Generates embeddings using:
    - **BERT**: Uses Sentence-BERT (`all-MiniLM-L6-v2`) for cosine similarity.
    - **USE**: Uses Universal Sentence Encoder for inner product similarity.
  - Combines the similarity scores by **averaging** them.
  - Re-ranks the documents based on the new combined scores.
  - Outputs the final results in **TREC format** to `Reranked_Results.txt`.

### **4. Output Generation**

Results of BM25 and reranked
- **Results File (`Results.txt`):**  
  - Format (TREC standard):  
    ```
    query_id Q0 doc_id rank score run_name
    ```
  - Example output:
    ```
1 Q0 42421723 1 0.2703 neural_rerank
1 Q0 21456232 2 0.2054 neural_rerank
1 Q0 7581911 3 0.1882 neural_rerank
    ```

- **Vocabulary Statistics & Sample Output:**  
  - The program prints:
    - The **vocabulary size** (number of unique terms).
    - A **sample of 100 tokens** from the vocabulary.
    - The **top 10 retrieved documents** for the first two queries.

---

## **How to Run the Program**
1. **Ensure the dataset files are in the working directory**  
   - `scifact/corpus.jsonl` (contains document collection)  
   - `scifact/queries.jsonl` (contains test queries) 

2. **Run the script using Python:**

 *assumption is that pip is already installed on the machine*
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

### Results

We tested two of the retrieval methods: 
BM25: A commonly used lexical-based ranking algorithm.
Neural re-ranking: This approach uses Sentence-BERT (all-MiniLM-L6-v2) and Universal Sentence Encoder (USE) to re-rank the top 100 results from BM25 based on semantic similarity. The results indicate that BM25 had better MAP scores, though the neural re-ranking performed well for the purposes of testing a more advanced retrieval system. Results.txt includes results from the best neural system.

### Evaluation 

We evaluated both BM25 and our neural reranking system (BERT + USE combined) using MAP and P@10 on the test queries.

| Method                | MAP     | P@10   |
|----------------------|---------|--------|
| BM25                 | 0.6357  | 0.0948 |
| BERT + USE Re-Ranking | 0.5333  | 0.0948 |

**Best System:** 
While BM25 had superior performance according to MAP, we opted for the final choice of the neural reranking system because the assignment focus was on neural IR types, plus the neural reranking was still ranked well under P@10. Results are from the combined BERT USE neural system in the `Results.txt` file.

### How to Run `generate_results_and_evaluate.py`

This script has two functions:
1. Scores BM25 results and reranked neural output on the basis of MAP and P@10 on the test set.
2. Produces the final `Results.txt` file based only on the highest system test queries (neural re-ranking).


# Setup and run (Mac)

### 1: Create and activate a virtual environment
 python3 -m venv venv
source venv/bin/activate

### 2: Install dependencies
pip install -r requirements.txt
pip install pandas

### 3: Run the script 
python3 generate_results_and_evaluate.py




### Language and Libraries

The language used was **Python**, and the libraries used:
- BM25: For initial document ranking.

- Sentence-BERT: Neural sentence embeddings (cosine similarity).

- Universal Sentence Encoder (USE): Document and query embeddings (inner product similarity).

- NumPy: Efficient document scoring and ranking.

- NLTK: For text tokenization, stopword removal, and stemming.

- TREC format: Output results in the required format.
