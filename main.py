from data_fetcher import search_papers, convert_abstract_inverted_index_to_text
from processor import TextProcessor
from clusterer import ResearchClusterer

def main(tema, num_papers, num_clusters):
    # 1. Buscar papers sobre un tema
    print(f"Buscando papers sobre: {tema}...")
    try:
        papers = search_papers(tema, per_page=num_papers)
        if not papers:
            print("No se encontraron papers para el tema especificado.")
            return
    except Exception as e:
        print(f"Error al buscar papers: {e}")
        return

    # 2. Procesar los abstracts
    processor = TextProcessor()
    abstracts = [convert_abstract_inverted_index_to_text(paper['abstract']) for paper in papers]
    cleaned_abstracts, valid_indices = processor.clean_texts(abstracts)

    # Filtrar artículos problemáticos
    papers = [papers[i] for i in valid_indices]
    cleaned_abstracts = [cleaned_abstracts[i] for i in valid_indices]

    if not papers:
        print("Error: No hay artículos válidos después de la limpieza.")
        return

    # Continuar con el análisis
    print("\nResultados de la búsqueda:")
    for i, p in enumerate(papers):
        print(f"\nTítulo: {p['title']}")
        print(f"Año: {p['year']}")
        print(f"Abstract limpio: {cleaned_abstracts[i]}")

    # 4. Extraer palabras clave principales
    print("\nExtrayendo palabras clave principales...")
    top_keywords = processor.extract_top_keywords(cleaned_abstracts, top_n=5)
    for i, keywords in enumerate(top_keywords):
        print(f"\nPalabras clave del artículo '{papers[i]['title']}': {', '.join(keywords)}")

    # 5. Clustering
    print("\nAgrupando investigaciones por similitud temática...")
    try:
        clusterer = ResearchClusterer(n_clusters=num_clusters)
        labels = clusterer.cluster_abstracts(cleaned_abstracts)
    except Exception as e:
        print(f"Error durante el clustering: {e}")
        return

    # 6. Identificar temas principales de los clusters
    try:
        cluster_topics = clusterer.identify_cluster_topics(cleaned_abstracts, labels)
    except Exception as e:
        print(f"Error al identificar los temas principales de los clusters: {e}")
        return

    # 7. Mostrar resultados por grupos
    print("\nResultados del clustering:")
    for cluster_id in range(num_clusters):  # Mostrar qué hay en cada cluster
        print(f"\n--- CLUSTER {cluster_id} ---")
        print(f"Tema principal: {', '.join(cluster_topics[cluster_id])}")
        cluster_papers = [papers[i]['title'] for i in range(len(labels)) if labels[i] == cluster_id]
        for title in cluster_papers:
            print(f" - {title}")

    # 8. Analizar Gaps de Investigación
    print("\n" + "="*30)
    print(" ANALIZADOR DE VACÍOS (GAPS) ")
    print("="*30)
    
    import pandas as pd
    from gap_analyzer import GapAnalyzer
    
    # Creamos un pequeño dataframe para el análisis
    df_results = pd.DataFrame({'cluster': labels})
    analyzer = GapAnalyzer()
    gaps = analyzer.analyze_gaps(df_results, cluster_topics)
    
    for gap in gaps:
        print(f"\nTemas: {gap['topics']}")
        print(f"Cantidad de papers: {gap['count']}")
        print(f"Estado: {gap['status']}")

if __name__ == "__main__":
    # Valores predeterminados para pruebas locales
    main("Artificial Intelligence in Mental Health", 20, 5)