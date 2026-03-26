import umap
import plotly.express as px
import pandas as pd

class ResearchVisualizer:
    def create_clusters_map(self, embeddings, labels, papers):
        # 1. Reducir dimensiones de 384 a 2
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # 2. Preparar datos para el gráfico
        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'title': [p['title'] for p in papers],
            'year': [p['year'] for p in papers],
            'cluster': [f"Cluster {l}" for l in labels]
        })
        
        # 3. Crear gráfico interactivo con Plotly
        fig = px.scatter(
            df, x='x', y='y', 
            color='cluster', 
            hover_data=['title', 'year'],
            title="Mapa de Conocimiento: ResearchGap AI",
            template="plotly_dark"
        )
        
        return fig