import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os

# 1. Download dataset dari KaggleHub
path = kagglehub.dataset_download("spsayakpaul/arxiv-paper-abstracts")
print("Path to dataset files:", path)

# Cari file CSV yang berisi dataset (asumsi 1 file CSV di dalam path)
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in downloaded dataset.")

dataset_path = os.path.join(path, csv_files[0])
print(f"Using dataset: {dataset_path}")

# 2. Load dataset
data = pd.read_csv(dataset_path)

# Pastikan kolom 'abstract' ada
if 'summaries' not in data.columns:
    raise KeyError("Dataset must have an 'summaries' column.")

abstracts = data['summaries'].fillna('')  # Jika ada null, isi dengan string kosong

# 3. Preprocessing sederhana
abstracts = abstracts.str.lower()

# Simpan dataset yang sudah diproses ke dalam folder proyek
processed_data = data[[ 'titles', 'terms']].copy()
processed_data['abstract'] = abstracts  # abstrak yang sudah lowercased

processed_data.to_csv('app/data/processed_papers.csv', index=False)

# 4. TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

# 5. Apply LSA (SVD)
lsa = TruncatedSVD(n_components=300, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# 6. Save models
os.makedirs('app/model', exist_ok=True)
with open('app/model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('app/model/lsa_model.pkl', 'wb') as f:
    pickle.dump(lsa, f)

with open('app/model/document_vectors.pkl', 'wb') as f:
    pickle.dump(lsa_matrix, f)

# 7. Save the papers metadata (title + abstract) untuk retrieval nanti
os.makedirs('app/data', exist_ok=True)
data[['titles', 'summaries', 'terms']].to_csv('app/data/papers.csv', index=False)

print("Training completed and models saved!")
