import re
import string

from sklearn.base import BaseEstimator, TransformerMixin


def clean_text(text: str) -> str:
    """
    Melakukan pembersihan teks sederhana:
    - lowercase
    - menghapus angka
    - menghapus tanda baca
    - menghapus whitespace berlebih
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer untuk preprocessing batch teks.
    Bisa digunakan di pipeline TF-IDF.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(text) for text in X]
