import umap
import plotly.express as px
import pandas as pd

class ResearchVisualizer:
    def create_clusters_map(self, embeddings, labels, papers):
        """
        Creates an interactive 2D knowledge map using UMAP and Plotly.
        
        :param embeddings: High-dimensional embeddings of the abstracts.
        :param labels: Cluster labels assigned to each abstract.
        :param papers: List of dictionaries containing paper metadata (e.g., title, year).
        :return: A Plotly figure object representing the knowledge map.
        """
        # 1. Reduce dimensions from 384 to 2 using UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # 2. Prepare data for the plot
        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'title': [p['title'] for p in papers],
            'year': [p['year'] for p in papers],
            'cluster': [f"Cluster {l}" for l in labels]
        })
        
        # 3. Create an interactive scatter plot with Plotly
        fig = px.scatter(
            df, x='x', y='y', 
            color='cluster', 
            hover_data=['title', 'year'],
            title="Knowledge Map: ResearchGap AI",
            template="plotly_dark"
        )
        
        return fig