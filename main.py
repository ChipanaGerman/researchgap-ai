from data_fetcher import search_papers, convert_abstract_inverted_index_to_text
from processor import TextProcessor
from clusterer import ResearchClusterer

def main(topic, num_papers, num_clusters):
    """
    Main function to analyze research papers on a given topic.
    
    :param topic: The research topic to search for.
    :param num_papers: Number of papers to analyze.
    :param num_clusters: Number of clusters to create.
    """
    # 1. Search for papers on the given topic
    print(f"Searching for papers on: {topic}...")
    try:
        papers = search_papers(topic, per_page=num_papers)
        if not papers:
            print("No papers found for the specified topic.")
            return
    except Exception as e:
        print(f"Error while searching for papers: {e}")
        return

    # 2. Process abstracts
    processor = TextProcessor()
    abstracts = [convert_abstract_inverted_index_to_text(paper['abstract']) for paper in papers]
    cleaned_abstracts, valid_indices = processor.clean_texts(abstracts)

    # Filter problematic articles
    papers = [papers[i] for i in valid_indices]
    cleaned_abstracts = [cleaned_abstracts[i] for i in valid_indices]

    if not papers:
        print("Error: No valid articles after cleaning.")
        return

    # Continue with the analysis
    print("\nSearch results:")
    for i, p in enumerate(papers):
        print(f"\nTitle: {p['title']}")
        print(f"Year: {p['year']}")
        print(f"Cleaned Abstract: {cleaned_abstracts[i]}")

    # 4. Extract main keywords
    print("\nExtracting main keywords...")
    top_keywords = processor.extract_top_keywords(cleaned_abstracts, top_n=5)
    for i, keywords in enumerate(top_keywords):
        print(f"\nKeywords for the article '{papers[i]['title']}': {', '.join(keywords)}")

    # 5. Perform clustering
    print("\nClustering research papers by thematic similarity...")
    try:
        clusterer = ResearchClusterer(n_clusters=num_clusters)
        labels = clusterer.cluster_abstracts(cleaned_abstracts)
    except Exception as e:
        print(f"Error during clustering: {e}")
        return

    # 6. Identify main topics for each cluster
    try:
        cluster_topics = clusterer.identify_cluster_topics(cleaned_abstracts, labels)
    except Exception as e:
        print(f"Error while identifying main topics for clusters: {e}")
        return

    # 7. Display clustering results
    print("\nClustering results:")
    for cluster_id in range(num_clusters):  # Display the content of each cluster
        print(f"\n--- CLUSTER {cluster_id} ---")
        print(f"Main Topic: {', '.join(cluster_topics[cluster_id])}")
        cluster_papers = [papers[i]['title'] for i in range(len(labels)) if labels[i] == cluster_id]
        for title in cluster_papers:
            print(f" - {title}")

    # 8. Analyze research gaps
    print("\n" + "="*30)
    print(" RESEARCH GAP ANALYZER ")
    print("="*30)
    
    import pandas as pd
    from gap_analyzer import GapAnalyzer
    
    # Create a small DataFrame for analysis
    df_results = pd.DataFrame({'cluster': labels})
    analyzer = GapAnalyzer()
    gaps = analyzer.analyze_gaps(df_results, cluster_topics)
    
    for gap in gaps:
        print(f"\nTopics: {gap['topics']}")
        print(f"Number of papers: {gap['count']}")
        print(f"Status: {gap['status']}")

if __name__ == "__main__":
    # Default values for local testing
    main("Artificial Intelligence in Mental Health", 20, 5)