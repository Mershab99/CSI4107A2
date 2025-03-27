# CSI4107 Assignment 2
### Members
- Mershab Issadien (300027272)
  - Preprocessing corpus by lowercasing, removing punctuation, applying tokenization, removing stopwords and stemming.
  - BM25 IR retrieval system.
      - Building BM25 index and ranking documents based on similarity scores.
  - Neural Re-ranking using `Sentence-BERT` (`all-MiniLM-L6-v2`) and `Universal Sentence Encoder` (USE).
    - Reranking the BM25 results.
    - Generating embeddings for both quries and top 100 BM25 retrieved documents.
    - Calculating similarity scores using cosine similarity for BERT and inner product similarity for USE.
    - Combining the scores from both models and averaging them to get the final re-ranked results.
    - Writing the results in TREC format.
- Buchra ____ (XXXXXXX)


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

TODO: update this with results of BM25 and reranked
- **Results File (`Results.txt`):**  
  - Format:  
    ```
    query_id Q0 doc_id rank score run_name
    ```
  - Example output:
    ```
    1 Q0 4983 1 0.8032 run_name
    1 Q0 5836 2 0.7586 run_name
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

TODO

### Language and Libraries

The language used was **Python**, and the libraries used:
- BM25: For initial document ranking.

- Sentence-BERT: Neural sentence embeddings (cosine similarity).

- Universal Sentence Encoder (USE): Document and query embeddings (inner product similarity).

- NumPy: Efficient document scoring and ranking.

- NLTK: For text tokenization, stopword removal, and stemming.

- TREC format: Output results in the required format.
