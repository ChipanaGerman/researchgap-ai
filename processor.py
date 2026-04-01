import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

class TextProcessor:
    def __init__(self, language_model="en_core_web_sm"):
        """
        Initializes the spaCy language model for text processing.
        
        :param language_model: Name of the spaCy language model to load.
        """
        self.nlp = spacy.load(language_model)

    def extract_top_keywords(self, cleaned_texts, top_n=10):
        """
        Extracts the most important keywords from a set of cleaned texts using TfidfVectorizer.
        
        :param cleaned_texts: List of cleaned texts.
        :param top_n: Number of keywords to extract per text.
        :return: List of lists containing the most important keywords for each text.
        """
        # Configure the vectorizer to ignore common words and limit the vocabulary
        vectorizer = TfidfVectorizer(
            stop_words="english",  # Remove English stop-words
            max_features=1000,    # Limit the vocabulary to the 1000 most relevant words
            min_df=2              # Ignore words that appear in fewer than 2 documents
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
            # Handle the case of empty texts or invalid vocabulary
            return [["No keywords found"] for _ in cleaned_texts]

    def clean_texts(self, texts):
        """
        Cleans a list of texts by removing stop-words, punctuation, numbers,
        irrelevant words, and applying lemmatization.
        
        :param texts: List of texts (abstracts) to clean.
        :return: List of cleaned texts and a list of valid abstract indices.
        """
        cleaned_texts = []
        valid_indices = []  # List to track indices of valid abstracts
        for i, text in enumerate(texts):
            if not text.strip():  # Ignore empty texts
                cleaned_texts.append("No abstract available")
                continue
            doc = self.nlp(text)
            cleaned = [
                token.lemma_ for token in doc
                if not token.is_stop  # Remove stop-words
                and not token.is_punct  # Remove punctuation
                and not token.is_space  # Remove spaces
                and not token.like_num  # Remove numbers
                and len(token) > 2  # Ignore very short words
                and re.match(r"^[a-zA-Z]+$", token.text)  # Only alphabetic words
            ]
            if cleaned:
                cleaned_texts.append(" ".join(cleaned).lower())
                valid_indices.append(i)  # Save the index of the valid abstract
            else:
                cleaned_texts.append("No abstract available")
        return cleaned_texts, valid_indices