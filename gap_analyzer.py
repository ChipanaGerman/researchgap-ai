import pandas as pd

class GapAnalyzer:
    def analyze_gaps(self, cluster_results, cluster_topics):
        """
        Analiza qué clusters tienen menos investigación.
        cluster_results: Diccionario o DataFrame con los títulos y su cluster ID.
        """
        # Contar cuántos papers hay por cluster
        counts = cluster_results['cluster'].value_counts().to_dict()
        
        analysis = []
        for cluster_id, count in counts.items():
            topics = cluster_topics.get(cluster_id, ["Unknown"])
            
            # Lógica simple de Gap: Menos papers = Mayor oportunidad
            status = "Saturado" if count > 7 else "Oportunidad (Gap)"
            if count <= 3:
                status = "Gap Crítico (Muy poca investigación)"
                
            analysis.append({
                'cluster': cluster_id,
                'count': count,
                'topics': ", ".join(topics),
                'status': status
            })
            
        return sorted(analysis, key=lambda x: x['count'])