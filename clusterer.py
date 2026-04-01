from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ResearchClusterer:
    def __init__(self, model_name="all-MiniLM-L6-v2", n_clusters=5):
        """
        Initializes the embedding model and the number of clusters.
        
        :param model_name: Name of the SentenceTransformer model.
        :param n_clusters: Number of clusters for KMeans.
        """
        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

    def cluster_abstracts(self, abstracts):
        """
        Converts abstracts into embeddings and clusters them using KMeans.
        
        :param abstracts: List of abstracts.
        :return: List of cluster labels for each abstract.
        """
        if not abstracts or not any(abstracts):
            raise ValueError("The list of abstracts is empty or contains no valid texts.")
        
        # Generate embeddings
        embeddings = self.model.encode(abstracts)
        
        # Perform clustering with KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        self.cluster_centers = kmeans.cluster_centers_
        return labels

    def identify_cluster_topics(self, abstracts, labels):
        """
        Identifies the main topics for each cluster based on the abstracts.
        
        :param abstracts: List of abstracts.
        :param labels: List of cluster labels for each abstract.
        :return: Dictionary mapping each cluster to its main topics.
        """
        if not abstracts or labels is None:
            raise ValueError("Abstracts or labels are empty.")
        
        # Group abstracts by cluster
        cluster_texts = {i: [] for i in range(self.n_clusters)}
        for abstract, label in zip(abstracts, labels):
            if abstract.strip():
                cluster_texts[label].append(abstract)
        
        cluster_topics = {}
        for cluster, texts in cluster_texts.items():
            if not texts:
                cluster_topics[cluster] = ["No topics found"]
                continue
            
            # Use TfidfVectorizer to find unique words for this cluster
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Sum TF-IDF scores for each word
                sums = tfidf_matrix.sum(axis=0)
                data = []
                for col, term in enumerate(feature_names):
                    data.append((term, sums[0, col]))
                
                # Sort and select the top 5 words
                ranking = sorted(data, key=lambda x: x[1], reverse=True)
                cluster_topics[cluster] = [t[0] for t in ranking[:5]]
            except:
                cluster_topics[cluster] = ["Generic/Misc"]
        
        return cluster_topics