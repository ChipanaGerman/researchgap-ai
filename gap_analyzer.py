import pandas as pd

class GapAnalyzer:
    def analyze_gaps(self, cluster_results, cluster_topics):
        """
        Analyzes which clusters have less research activity (gaps).
        
        :param cluster_results: Dictionary or DataFrame containing titles and their cluster IDs.
        :param cluster_topics: Dictionary mapping cluster IDs to their main topics.
        :return: A sorted list of dictionaries with gap analysis results.
        """
        # Count the number of papers per cluster
        counts = cluster_results['cluster'].value_counts().to_dict()
        
        analysis = []
        for cluster_id, count in counts.items():
            topics = cluster_topics.get(cluster_id, ["Unknown"])
            
            # Simple gap logic: Fewer papers = Greater opportunity
            status = "Saturated" if count > 7 else "Opportunity (Gap)"
            if count <= 3:
                status = "Critical Gap (Very little research)"
                
            analysis.append({
                'cluster': cluster_id,
                'count': count,
                'topics': ", ".join(topics),
                'status': status
            })
            
        # Sort the analysis by the number of papers in ascending order
        return sorted(analysis, key=lambda x: x['count'])