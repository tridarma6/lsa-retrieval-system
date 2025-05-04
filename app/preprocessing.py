from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Unduh resource NLTK ---
nltk.download('stopwords')

# --- Inisialisasi tokenizer, stopwords, stemmer, dan juga flask ---
app = Flask(__name__)
CORS(app)
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Fungsi Preprocessing ---
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

# --- Load Data dari CSV (50 baris) dan siapkan data saat server akan start ---
file_path = "papers_dataset.csv"
df = pd.read_csv(file_path, nrows=50)
df = df.dropna(subset=['summaries'])
documents = df['summaries'].astype(str).tolist()

# --- Preprocessing Semua Baris Abstract ---
tokenized_documents = [preprocess(doc) for doc in documents]

# # # --- Tampilkan Hasil Preprocessing 5 Baris Abstract ---
# for i, doc in enumerate(tokenized_documents[:5], 1):
#     print(f"Abstract {i}:\n{doc}\n")

# --- Gabungkan token jadi teks untuk setiap dokumen ---
cleaned_documents = [' '.join(doc) for doc in tokenized_documents]

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_documents)


# End Point API
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("query", "")
    
    # Preprocess & TF-IDF query
    preprocessed_query = preprocess(query)
    query_string = ' '.join(preprocessed_query)
    query_vector = vectorizer.transform([query_string])
    
    # Cosine Similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df['Similarity Score'] = similarities
    df_result = df.sort_values(by='Similarity Score', ascending=False)
    
    # Ambil Top 5 dan ubah ke JSON
    top_5 = df_result[['titles', 'Similarity Score']].head(5)
    results = top_5.to_dict(orient='records')

    return jsonify({"query": query, "results": results})


# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)


# # --- Input Query dari Pengguna ---
# query = input("Masukkan query pencarian berupa abstract: ")
# preprocessed_query = preprocess(query)
# query_string = ' '.join(preprocessed_query)
# query_vector = vectorizer.transform([query_string])

# # --- Hitung Cosine Similarity antara Query dan Abtract ---
# similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# # --- Tampilkan Top 5 Abstract Paling Relevan Dengan Query dan Hasil Paling Mirip ---
# df['Similarity Score'] = similarities
# df_result = df.sort_values(by='Similarity Score', ascending=False)

# print(f"\nQuery: {query}")
# print("\nTop 5 Abstract Paling Relevan Dengan Query:")
# print(df_result[['titles', 'Similarity Score']].head(5))    
