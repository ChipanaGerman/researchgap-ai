from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ResearchClusterer:
    def __init__(self, model_name="all-MiniLM-L6-v2", n_clusters=5):
        """
        Inicializa el modelo de embeddings y el número de clusters.
        
        :param model_name: Nombre del modelo de SentenceTransformer.
        :param n_clusters: Número de clusters para KMeans.
        """
        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

    def cluster_abstracts(self, abstracts):
        """
        Convierte los abstracts en embeddings y los agrupa en clusters.
        
        :param abstracts: Lista de abstracts.
        :return: Lista de etiquetas de clusters para cada abstract.
        """
        if not abstracts or not any(abstracts):
            raise ValueError("La lista de abstracts está vacía o no contiene textos válidos.")
        
        # Generar embeddings
        embeddings = self.model.encode(abstracts)
        
        # Agrupar con KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        self.cluster_centers = kmeans.cluster_centers_
        return labels

    def identify_cluster_topics(self, abstracts, labels):
        if not abstracts or labels is None:
            raise ValueError("Los abstracts o las etiquetas están vacíos.")
        
        cluster_texts = {i: [] for i in range(self.n_clusters)}
        for abstract, label in zip(abstracts, labels):
            if abstract.strip():
                cluster_texts[label].append(abstract)
        
        cluster_topics = {}
        for cluster, texts in cluster_texts.items():
            if not texts:
                cluster_topics[cluster] = ["No topics found"]
                continue
            
            # Usamos TfidfVectorizer para encontrar palabras únicas de este cluster
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Sumamos los scores de TF-IDF para cada palabra
                sums = tfidf_matrix.sum(axis=0)
                data = []
                for col, term in enumerate(feature_names):
                    data.append((term, sums[0, col]))
                
                # Ordenamos y sacamos las 5 mejores
                ranking = sorted(data, key=lambda x: x[1], reverse=True)
                cluster_topics[cluster] = [t[0] for t in ranking[:5]]
            except:
                cluster_topics[cluster] = ["Generic/Misc"]
        
        return cluster_topics