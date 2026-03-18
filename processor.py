import spacy

class TextProcessor:
    def __init__(self, language_model="en_core_web_sm"):
        # Cargar el modelo de spaCy
        self.nlp = spacy.load(language_model)

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