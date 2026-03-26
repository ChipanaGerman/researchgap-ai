import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

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
        # Configurar el vectorizador para ignorar palabras comunes y limitar el vocabulario
        vectorizer = TfidfVectorizer(
            stop_words="english",  # Eliminar stop-words en inglés
            max_features=1000,    # Limitar el vocabulario a las 1000 palabras más relevantes
            min_df=2              # Ignorar palabras que aparecen en menos de 2 documentos
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            top_keywords = []
            for row in tfidf_matrix:
                sorted_indices = np.argsort(row.toarray()[0])[::-1]
                top_features = feature_names[sorted_indices[:top_n]]
                top_keywords.append(top_features.tolist())
            
            return top_keywords
        except ValueError:
            # Manejar el caso de textos vacíos o sin vocabulario válido
            return [["No keywords found"] for _ in cleaned_texts]

    def clean_texts(self, texts):
        """
        Limpia una lista de textos eliminando stop-words, puntuación, números,
        palabras irrelevantes y aplicando lematización.
        
        :param texts: Lista de textos (abstracts) a limpiar.
        :return: Lista de textos limpios y una lista de índices de abstracts válidos.
        """
        cleaned_texts = []
        valid_indices = []  # Lista para rastrear los índices de abstracts válidos
        for i, text in enumerate(texts):
            if not text.strip():  # Ignorar textos vacíos
                cleaned_texts.append("No abstract available")
                continue
            doc = self.nlp(text)
            cleaned = [
                token.lemma_ for token in doc
                if not token.is_stop  # Eliminar stop-words
                and not token.is_punct  # Eliminar puntuación
                and not token.is_space  # Eliminar espacios
                and not token.like_num  # Eliminar números
                and len(token) > 2  # Ignorar palabras muy cortas
                and re.match(r"^[a-zA-Z]+$", token.text)  # Solo palabras alfabéticas
            ]
            if cleaned:
                cleaned_texts.append(" ".join(cleaned).lower())
                valid_indices.append(i)  # Guardar el índice del abstract válido
            else:
                cleaned_texts.append("No abstract available")
        return cleaned_texts, valid_indices