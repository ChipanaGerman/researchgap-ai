from data_fetcher import search_papers, convert_abstract_inverted_index_to_text
from processor import TextProcessor
from clusterer import ResearchClusterer

def main():
    # 1. Buscar papers sobre un tema
    tema = "Artificial Intelligence in Mental Health"
    print(f"Buscando papers sobre: {tema}...")
    try:
        papers = search_papers(tema)
        if not papers:
            print("No se encontraron papers para el tema especificado.")
            return
    except Exception as e:
        print(f"Error al buscar papers: {e}")
        return

    # 2. Procesar los abstracts
    processor = TextProcessor()
    abstracts = [convert_abstract_inverted_index_to_text(p['abstract']) for p in papers]
    cleaned_abstracts = processor.clean_texts(abstracts)

    if not any(cleaned_abstracts):
        print("Error: No hay abstracts válidos después de la limpieza.")
        return

    # 3. Mostrar resultados limpios y extraer palabras clave
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
        clusterer = ResearchClusterer(n_clusters=5)
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
    for cluster_id in range(5):  # Mostrar qué hay en cada cluster
        print(f"\n--- CLUSTER {cluster_id} ---")
        print(f"Tema principal: {', '.join(cluster_topics[cluster_id])}")
        cluster_papers = [papers[i]['title'] for i in range(len(labels)) if labels[i] == cluster_id]
        for title in cluster_papers:
            print(f" - {title}")

if __name__ == "__main__":
    main()