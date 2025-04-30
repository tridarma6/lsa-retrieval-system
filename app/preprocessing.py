import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Unduh resource NLTK ---
nltk.download('stopwords')

# --- Inisialisasi tokenizer, stopwords, dan stemmer ---
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Fungsi Preprocessing ---
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

# --- Load Data dari CSV (50 baris) ---
file_path = "papers_dataset.csv"
df = pd.read_csv(file_path, nrows=50)
df = df.dropna(subset=['summaries'])
documents = df['summaries'].astype(str).tolist()

# --- Preprocessing Semua Baris Abstract ---
tokenized_documents = [preprocess(doc) for doc in documents]

# # --- Tampilkan Hasil Preprocessing 5 Baris Abstract ---
for i, doc in enumerate(tokenized_documents[:5], 1):
    print(f"Abstract {i}:\n{doc}\n")

# # --- Gabungkan token jadi teks untuk setiap dokumen ---
# cleaned_documents = [' '.join(doc) for doc in tokenized_documents]

# # --- TF-IDF Vectorization ---
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(cleaned_documents)

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
