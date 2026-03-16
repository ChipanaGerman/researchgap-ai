import requests
def convert_abstract_inverted_index_to_text(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""
    words = []
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            words.append((position, word))
    words.sort()  # Sort by position
    return " ".join(word for _, word in words)
def search_papers(keyword):
    base_url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{keyword}",
        "per_page": 10
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    
    data = response.json()
    papers = []
    for item in data.get('results', []):
        paper = {
            "title": item.get("title"),
            "year": item.get("publication_year"),
            "abstract": item.get("abstract_inverted_index"),
            "link": item.get("id")
        }
        papers.append(paper)
    
    return papers