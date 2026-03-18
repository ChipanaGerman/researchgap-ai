import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextProcessor:
    def __init__(self, language_model="en_core_web_sm"):
        # Cargar el modelo de spaCy
        self.nlp = spacy.load(language_model)

    def extract_top_keywords(self, cleaned_texts, top_n=10):
            """
            Extrae las palabras más importantes de un conjunto de textos limpios 
            utilizando TfidfVectorizer.
            
            :param cleaned_texts: Lista de textos limpios.
            :param top_n: Número de palabras clave a extraer.
            :return: Lista de listas con las palabras clave más importantes por texto.
            """
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            top_keywords = []
            for row in tfidf_matrix:
                sorted_indices = np.argsort(row.toarray()[0])[::-1]
                top_features = feature_names[sorted_indices[:top_n]]
                top_keywords.append(top_features.tolist())
            
            return top_keywords
    
    def clean_texts(self, texts):
        """
        Limpia una lista de textos eliminando stop-words, puntuación, 
        convirtiendo a minúsculas y aplicando lematización.
        
        :param texts: Lista de textos (abstracts) a limpiar.
        :return: Lista de textos limpios.
        """
        cleaned_texts = []
        for text in texts:
            doc = self.nlp(text)
            cleaned = [
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            cleaned_texts.append(" ".join(cleaned).lower())
        return cleaned_texts