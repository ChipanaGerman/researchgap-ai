import requests

def convert_abstract_inverted_index_to_text(abstract_inverted_index):
    """
    Converts an inverted index of an abstract into plain text.
    
    :param abstract_inverted_index: Dictionary with words as keys and positions as values.
    :return: Plain text reconstructed from the inverted index.
    """
    if not abstract_inverted_index:
        return ""
    words = []
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            words.append((position, word))
    words.sort()  # Sort by position
    return " ".join(word for _, word in words)

def search_papers(keyword, per_page=10, max_results=100):
    """
    Searches for articles in the OpenAlex API using a keyword.
    
    :param keyword: Keyword to search for in article titles.
    :param per_page: Number of results per page (default: 10).
    :param max_results: Maximum number of results to retrieve (default: 100).
    :return: List of dictionaries containing article information.
    """
    base_url = "https://api.openalex.org/works"
    papers = []
    page = 1  # Initial page

    # Dynamically adjust max_results if it is smaller than per_page
    max_results = min(max_results, per_page)

    while len(papers) < max_results:
        params = {
            "filter": f"title.search:{keyword}",
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an error if the status code is not 200
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error while making the API request: {e}")
        
        try:
            data = response.json()
        except ValueError:
            raise Exception("Error decoding the API response JSON.")
        
        results = data.get('results', [])
        if not results:  # Exit the loop if there are no more results
            break
        
        for item in results:
            # Validate that the required fields exist
            if not item.get("title") or not item.get("abstract_inverted_index"):
                continue
            paper = {
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "abstract": item.get("abstract_inverted_index"),
                "link": item.get("id")
            }
            papers.append(paper)
            if len(papers) >= max_results:  # Stop if the limit is reached
                break
        
        page += 1  # Move to the next page
    
    return papers