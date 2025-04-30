import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Unduh resource NLTK ---
nltk.download('stopwords')

# --- Inisialisasi tokenizer, stopwords, dan stemmer ---
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = StemmerFactory().create_stemmer()

# --- Fungsi Preprocessing ---
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

# --- Load Data dari CSV (50 baris) ---
file_path = "arxiv_data.csv"
df = pd.read_csv(file_path, nrows=50)
df = df.dropna(subset=['summaries'])
documents = df['summaries'].astype(str).tolist()

# --- Preprocessing Semua Dokumen ---
tokenized_documents = [preprocess(doc) for doc in documents]

# --- Tampilkan Hasil Preprocessing Contoh 5 Dokumen ---
for i, doc in enumerate(tokenized_documents[:5], 1):
    print(f"Dokumen {i}:\n{doc}\n")
