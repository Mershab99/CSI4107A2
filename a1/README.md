# CSI4107 Assignment 1
### Members
- Mershab Issadien (300027272)
  - Solo project group, therefore I completed every aspect of the assignment myself.


### **Functionality of the Program**

This program implements an **Information Retrieval (IR) system** using the **BM25 ranking model**. It processes a collection of scientific documents (from `corpus.jsonl`), indexes them, and ranks them against a set of test queries (from `queries.jsonl`). The results are saved in the required TREC format (`Results.txt`). The key components of the system are **preprocessing, indexing, retrieval, and evaluation**.

---

## **Functionality Overview**
### **1. Preprocessing**
- **Text Normalization:** Converts text to lowercase.
- **Punctuation Removal:** Removes special characters and numbers to focus on meaningful words.
- **Tokenization:** Splits text into words using NLTK’s `word_tokenize()`.
- **Stopword Removal:** Uses a predefined list of stopwords (customizable).
- **Stemming:** Uses the Porter Stemmer to reduce words to their root forms.
- **Processing Scope:** Both the **title** and **text** fields of each document are combined for indexing.

### **2. Indexing**
- **Inverted Index Construction:** Stores document IDs associated with tokenized words.
- **BM25 Indexing:** Uses the `rank_bm25` library to build a **BM25Okapi** index, which helps in computing similarity scores.

### **3. Retrieval & Ranking**
- **Query Processing:** Test queries are preprocessed using the same steps as documents.
- **Similarity Computation:** BM25 scores are calculated for each document with respect to each query.
- **Ranking:** Documents are sorted based on BM25 scores, and only the **top 100** results per query are stored.

### **4. Output Generation**
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
Vocabulary Size: 40229
Sample 100 Tokens: ['020085', 'tigr', 'condomsl', 'parasitederiv', 'irinduc', 'pyreflac', 'microbicid', 'anticd19', '11000', 'intrapancreat', 'paglialunga', '043087', 'earlygener', 'limbal', 'naipnlrc4', 'asyet', 'twentynin', 'intraatri', 'puff', 'valproic', 'time', '105personyear', 'magnet', 'guerbet', 'power', 'datane', 'ev', 'dnabound', 'wlc', '206', 'model', '1met', 'y27632', 'fingerprint', 'agonistdepend', 'su', '00471', '4r3', '1227', 'bloodpressurelow', 'o6methylguaninedna', 'ci101', '4e', 'pho85p', 'blowout', 'leukocidin', 'rscmediat', 'archaeaoeukaryot', 'kccq', '1947august', 'readdress', 'secondlin', 'thermogen', 'unexplain', 'endothelium', 'mispercept', 'taurolithochol', 'propeptid', 'oastimul', 'dens', 'positionalclon', 'oncolysi', 'int96', 'sitosteroltocholesterol', 'icusricu', 'h3n2vm', 'peristom', '904', 'ctlmediat', 'designmethodologyapproach', 'noninstitution', 'cxcl10', 'virusspecif', 'chromosome21specif', 'gilt', 'afm', 'ecmdegrad', 'hh', 'clickhal', 'ketamin', 'zp', 'koba', 'lobectomi', 'um', 'durninglawr', 'nonsepsi', 'disast', 'gnp', 'ncare', 'vadilex', 'ccnd1ncrna', 'nk', '6800', 'resurg', 'lipman', '0588', 'turc', 'genomicsproteom', 'faksrc', 'patterninclud']

First 10 Answers for First 2 Queries:
Query 9:
  Rank 1: Doc 44265107, Score 47.5456
  Rank 2: Doc 25182647, Score 18.5641
  Rank 3: Doc 24700152, Score 17.7919
  Rank 4: Doc 21186109, Score 14.9019
  Rank 5: Doc 14647747, Score 14.7910
  Rank 6: Doc 16737210, Score 14.7152
  Rank 7: Doc 8190282, Score 14.4329
  Rank 8: Doc 21859699, Score 14.3579
  Rank 9: Doc 37699461, Score 14.2789
  Rank 10: Doc 24916604, Score 14.1482
Query 11:
  Rank 1: Doc 25510546, Score 39.4514
  Rank 2: Doc 20904154, Score 33.6915
  Rank 3: Doc 13780287, Score 33.6728
  Rank 4: Doc 4399311, Score 32.7990
  Rank 5: Doc 29459383, Score 32.4992
  Rank 6: Doc 7482674, Score 32.3748
  Rank 7: Doc 32587939, Score 31.8743
  Rank 8: Doc 19708993, Score 29.6165
  Rank 9: Doc 13958154, Score 29.0490
  Rank 10: Doc 8453819, Score 28.7175


### Language and Libraries

The language used was **Python** and the libraries used are in the `requirements.txt` file. Including the `rank-bm25` as well as `nltk` libraries which provided much of the helper methods.