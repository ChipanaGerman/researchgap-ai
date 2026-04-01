import streamlit as st
import pandas as pd
import plotly.express as px
import umap
from data_fetcher import search_papers, convert_abstract_inverted_index_to_text
from processor import TextProcessor
from clusterer import ResearchClusterer
from gap_analyzer import GapAnalyzer

# Page configuration
st.set_page_config(page_title="ResearchGap AI", page_icon="🧬", layout="wide")

# Title and description
st.title("🧬 ResearchGap AI: Research Gap Finder")
st.markdown("Automatically analyze thousands of papers to find opportunities for theses or startups.")

# --- Sidebar Configuration ---
st.sidebar.header("Settings")
query = st.sidebar.text_input("🔍 Research topic:", "Artificial Intelligence in Mental Health")
num_papers = st.sidebar.slider("📄 Number of papers to analyze:", 10, 100, 20)  # Adjust range as needed
num_clusters = st.sidebar.slider("🔢 Number of topics (clusters):", 2, 10, 3)

# --- Action Button ---
if st.sidebar.button("Analyze"):
    with st.spinner("Fetching papers and processing data..."):
        # 1. Fetch data
        papers = search_papers(query, per_page=num_papers)
        
        if not papers:
            st.error("No results found.")
        else:
            # 2. Process abstracts
            processor = TextProcessor()
            abstracts = [convert_abstract_inverted_index_to_text(p['abstract']) for p in papers]
            cleaned_abstracts, valid_indices = processor.clean_texts(abstracts)
            
            # Filter problematic articles
            papers = [papers[i] for i in valid_indices]
            cleaned_abstracts = [cleaned_abstracts[i] for i in valid_indices]
            
            if not cleaned_abstracts:
                st.error("No valid abstracts found after cleaning.")
                st.stop()
            
            # Validate that n_clusters <= n_samples
            n_clusters = min(num_clusters, len(cleaned_abstracts))
            if n_clusters < 2:
                st.error("Not enough valid abstracts to perform clustering.")
                st.stop()
            
            # 3. Perform clustering
            clusterer = ResearchClusterer(n_clusters=n_clusters)
            labels = clusterer.cluster_abstracts(cleaned_abstracts)
            cluster_topics = clusterer.identify_cluster_topics(cleaned_abstracts, labels)
            
            # 4. Analyze research gaps
            df_results = pd.DataFrame({'cluster': labels})
            analyzer = GapAnalyzer()
            gaps = analyzer.analyze_gaps(df_results, cluster_topics)
            
            # --- Display Results ---
            
            # Column 1: Metrics and Gaps
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Detected Opportunities")
                for gap in gaps:
                    with st.expander(f"📌 {gap['status']}"):
                        st.write(f"**Topics:** {gap['topics']}")
                        st.write(f"**Current articles:** {gap['count']}")
            
            # Column 2: Knowledge Map Visualization
            with col2:
                st.subheader("Knowledge Map")
                # Reduce dimensions with UMAP (this may take some time)
                embeddings = clusterer.model.encode(cleaned_abstracts)
                reducer = umap.UMAP(n_neighbors=min(15, num_papers-1), min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings)
                
                viz_df = pd.DataFrame({
                    'x': embedding_2d[:, 0],
                    'y': embedding_2d[:, 1],
                    'Title': [p['title'] for p in papers],
                    'Cluster': [f"Topic {l}" for l in labels]
                })
                
                fig = px.scatter(viz_df, x='x', y='y', color='Cluster', 
                                hover_data=['Title'], template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            # --- Data Table ---
            st.divider()
            st.subheader("Research Details")
            st.dataframe(pd.DataFrame(papers)[['title', 'year', 'link']])