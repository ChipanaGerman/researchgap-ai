import streamlit as st
import pandas as pd
import plotly.express as px
import umap
from data_fetcher import search_papers, convert_abstract_inverted_index_to_text
from processor import TextProcessor
from clusterer import ResearchClusterer
from gap_analyzer import GapAnalyzer

# Configuración de la página
st.set_page_config(page_title="ResearchGap AI", page_icon="🧬", layout="wide")

st.title("🧬 ResearchGap AI: Buscador de Vacíos de Investigación")
st.markdown("Analiza automáticamente miles de papers para encontrar oportunidades de tesis o startups.")

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("Configuración")
query = st.sidebar.text_input("🔍 Tema de investigación:", "Artificial Intelligence in Mental Health")
num_papers = st.sidebar.slider("📄 Número de papers a analizar:", 10, 100, 20)  # Ajusta el rango si es necesario
num_clusters = st.sidebar.slider("🔢 Número de temas (clusters):", 2, 10, 3)

# --- Botón de Acción ---
if st.sidebar.button("Analizar"):
    with st.spinner("Buscando papers y procesando IA..."):
        # 1. Buscar Datos
        papers = search_papers(query, per_page=num_papers)
        
        if not papers:
            st.error("No se encontraron resultados.")
        else:
            # 2. Procesar Texto
            processor = TextProcessor()
            abstracts = [convert_abstract_inverted_index_to_text(p['abstract']) for p in papers]
            cleaned_abstracts, valid_indices = processor.clean_texts(abstracts)
            
            # Filtrar artículos problemáticos
            papers = [papers[i] for i in valid_indices]
            cleaned_abstracts = [cleaned_abstracts[i] for i in valid_indices]
            
            if not cleaned_abstracts:
                st.error("No hay abstracts válidos después de la limpieza.")
                st.stop()
            
            # Validar que n_clusters <= n_samples
            n_clusters = min(num_clusters, len(cleaned_abstracts))
            if n_clusters < 2:
                st.error("No hay suficientes abstracts válidos para realizar el clustering.")
                st.stop()
            
            # 3. Clustering
            clusterer = ResearchClusterer(n_clusters=n_clusters)
            labels = clusterer.cluster_abstracts(cleaned_abstracts)
            cluster_topics = clusterer.identify_cluster_topics(cleaned_abstracts, labels)
            
            # 4. Análisis de Gaps
            df_results = pd.DataFrame({'cluster': labels})
            analyzer = GapAnalyzer()
            gaps = analyzer.analyze_gaps(df_results, cluster_topics)
            
            # --- MOSTRAR RESULTADOS ---
            
            # Columna 1: Métricas y Gaps
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Oportunidades Detectadas")
                for gap in gaps:
                    with st.expander(f"📌 {gap['status']}"):
                        st.write(f"**Temas:** {gap['topics']}")
                        st.write(f"**Artículos actuales:** {gap['count']}")
            
            # Columna 2: Visualización Matemática
            with col2:
                st.subheader("Mapa del Conocimiento")
                # Reducir dimensiones con UMAP (esto tarda un poquito)
                embeddings = clusterer.model.encode(cleaned_abstracts)
                reducer = umap.UMAP(n_neighbors=min(15, num_papers-1), min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings)
                
                viz_df = pd.DataFrame({
                    'x': embedding_2d[:, 0],
                    'y': embedding_2d[:, 1],
                    'Título': [p['title'] for p in papers],
                    'Cluster': [f"Tema {l}" for l in labels]
                })
                
                fig = px.scatter(viz_df, x='x', y='y', color='Cluster', 
                                hover_data=['Título'], template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            # --- Tabla de Datos ---
            st.divider()
            st.subheader("Detalle de la Investigación")
            st.dataframe(pd.DataFrame(papers)[['title', 'year', 'link']])