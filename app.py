import os  # Importing the OS module for file system operations
import re  # Importing regular expressions for text processing
import math  # Importing math for log-based computations
import numpy as np  # Importing NumPy for vector and matrix operations
from collections import defaultdict  # Importing default dictionary for inverted index creation
from flask import Flask, render_template, request  # Importing Flask for web application

app = Flask(__name__)  # Initializing Flask app

# In-memory dictionary to store replies for each document
replies = defaultdict(list)  # Keys: document IDs, Values: list of replies


# 1. Loading documents from the specified folder
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):  # Listing files in the folder
        if filename.endswith('.txt'):  # Filtering to only text files
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                docs[filename] = file.read()  # Reading file content
    return docs  # Returning loaded documents


# 2. Preprocessing the text by tokenizing and lowercasing
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())  # Tokenizing and lowercasing the text


# 3. Creating an inverted index from the documents
def create_inverted_index(docs):
    inverted_index = defaultdict(list)  # Creating a dictionary for the inverted index
    for doc_id, content in docs.items():
        words = preprocess(content)  # Preprocessing document content
        for word in set(words):  # Using unique words
            inverted_index[word].append(doc_id)  # Appending document ID to the word in index
    return inverted_index  # Returning the inverted index


# 4. Calculating term frequency (TF) for a given term in a document
def term_frequency(term, document):
    if len(document) == 0:  # Preventing division by zero
        return 0
    return document.count(term) / len(document)  # Calculating the frequency of the term


# 5. Calculating inverse document frequency (IDF) for a given term across documents
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)  # Counting documents containing the term
    return math.log(len(all_documents) / (1 + num_docs_containing_term))  # Calculating IDF with smoothing


# 6. Computing the TF-IDF vector for a document based on the vocabulary
def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:  # Iterating through all terms in the vocabulary
        tf = term_frequency(term, document)  # Calculating TF
        idf = inverse_document_frequency(term, all_documents)  # Calculating IDF
        tfidf_vector.append(tf * idf)  # Appending the TF-IDF score
    return np.array(tfidf_vector)  # Returning TF-IDF vector as NumPy array


# 7. Calculating cosine similarity between two TF-IDF vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)  # Calculating dot product of vectors
    norm_vec1 = np.linalg.norm(vec1)  # Calculating norm of the first vector
    norm_vec2 = np.linalg.norm(vec2)  # Calculating norm of the second vector
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 and norm_vec2 else 0.0  # Calculating cosine similarity


# 8. Retrieving and ranking documents based on cosine similarity with the query
def retrieve_documents(query, inverted_index, docs, vocab):
    query_terms = preprocess(query)  # Preprocessing the query

    if not query_terms:  # If query is empty, return no results
        return []

    # Step 1: Compute TF-IDF vector for the query
    query_vector = compute_tfidf(query_terms, list(docs.values()), vocab)  # Vectorizing the query

    similarities = {}  # Dictionary to store similarity scores

    # Step 2: For each document, compute its TF-IDF vector and calculate similarity with query
    for doc_id, content in docs.items():
        doc_vector = compute_tfidf(preprocess(content), list(docs.values()), vocab)  # Vectorizing the document
        similarity_score = cosine_similarity(query_vector, doc_vector)  # Computing cosine similarity

        similarities[doc_id] = similarity_score  # Storing the similarity score

    # Step 3: Rank the documents based on their similarity scores in descending order
    ranked_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)  # Sorting by similarity score

    # Step 4: Return the ranked results (document ID and similarity score)
    return ranked_docs


# Ground truth for relevant documents (as in the query)
relevances = {
    "What are common causes for battery life deterioration after a software update on Android/iOS?": {
        "Andriodupdate.txt", "huwaeiissue.txt", "ios16batteryissue.txt", "ios17issue.txt", "iosbatteryissue.txt",
        "iosissue.txt", "oneplusissue.txt", "pixelbatteryissue.txt", "samsungphoneissue.txt"
    },
    "How to troubleshoot poor battery performance on Android 14/iOS 17?": {
        "Andriodupdate.txt", "huwaeiissue.txt", "ios17issue.txt", "oneplusissue.txt", "samsungphoneissue.txt"
    },
    "Is there a known bug causing battery drain after the latest iOS/Android update?": {
        "Andriodupdate.txt", "huwaeiissue.txt", "ios16batteryissue.txt", "ios17issue.txt", "iosbatteryissue.txt",
        "iosissue.txt", "oneplusissue.txt", "pixelbatteryissue.txt", "samsungphoneissue.txt"
    },
    "How can I improve battery life after upgrading to the latest iOS/Android version?": {
        "Andriodupdate.txt", "huwaeiissue.txt", "ios17issue.txt", "oneplusissue.txt", "samsungphoneissue.txt"
    },
    "Why is my laptop restarting randomly after a Windows update?": {
        "Laptopissue.txt"
    },
    "How can I fix random laptop restarts following a Windows update?": {
        "Laptopissue.txt"
    },
    "Is the latest Windows update causing system instability and restarts?": {
        "Laptopissue.txt"
    },
    "What are troubleshooting steps for preventing random shutdowns after a Windows update?": {
        "Laptopissue.txt"
    },
    "What should I do if a package is delayed and tracking information is not updated?": {
        "consoleissue.txt", "phonedeliveryissue.txt"
    },
    "How to escalate shipping delays for online orders?": {
        "consoleissue.txt", "phonedeliveryissue.txt"
    },
    "How can I request a refund for a delayed or lost package?": {
        "consoleissue.txt", "phonedeliveryissue.txt"
    },
    "Why does my Wi-Fi keep disconnecting multiple times a day?": {
        "wifiissue.txt"
    },
    "What troubleshooting steps can fix frequent Wi-Fi disconnections?": {
        "wifiissue.txt"
    },
    "How can I fix my smart TV disconnecting from Wi-Fi after a software update?": {
        "smartTvissue.txt"
    },
    "Why is my fitness tracker giving inaccurate readings after a recent update?": {
        "fitnesstrackerissue.txt"
    },
    "How to troubleshoot syncing issues between a fitness tracker and its app?": {
        "fitnesstrackerissue.txt"
    },
    "Why is my phone's battery draining faster after the latest Android/iOS update?": {
        "Andriodupdate.txt", "huwaeiissue.txt", "ios16batteryissue.txt", "ios17issue.txt", "iosbatteryissue.txt",
        "iosissue.txt", "oneplusissue.txt", "pixelbatteryissue.txt", "samsungphoneissue.txt"
    },
    "How can I extend battery life after updating my Android/iPhone?": {
        "Andriodupdate.txt", "huwaeiissue.txt", "ios17issue.txt", "oneplusissue.txt", "samsungphoneissue.txt"
    }
}


# Evaluation Metric: Precision at K
def precision_at_k(retrieved_docs, relevant_docs, k):
    if k == 0:  # Prevent division by zero if k is 0
        return 0.0
    retrieved_at_k = [doc for doc, _ in retrieved_docs[:k]]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
    return len(relevant_retrieved) / k  # Safe to divide


# 9. Loading documents and building the inverted index
folder_path = 'techsupport'  # Your folder path here
docs = load_documents(folder_path)  # Loading documents
inverted_index = create_inverted_index(docs)  # Creating the inverted index
vocab = sorted(set(term for doc in docs.values() for term in preprocess(doc)))  # Creating sorted vocabulary


# 10. Rendering the index page
@app.route('/')
def index():
    return render_template('main.html')  # Rendering home page


# 11. Handling the search and displaying results
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']  # Getting the query from the form submission
    results = retrieve_documents(query, inverted_index, docs, vocab)  # Retrieving ranked results

    # Ground truth for current query
    relevant_docs = relevances.get(query, set())

    # Debugging: Print query, relevant documents, and top K results
    print(f"Query: {query}")
    print(f"Expected Relevant Documents: {relevant_docs}")
    print(f"Top Results: {[doc_id for doc_id, _ in results[:10]]}")  # Display top 10 results

    # Evaluation
    precision = precision_at_k(results, relevant_docs, 10)

    return render_template('resultspage.html', query=query, results=results, docs=docs, replies=replies,
                           precision=precision)  # Render results


if __name__ == '__main__':
    app.run(debug=True)  # Running the Flask app in debug mode
