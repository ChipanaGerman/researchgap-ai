import requests

def convert_abstract_inverted_index_to_text(abstract_inverted_index):
    """
    Convierte un índice invertido de un abstract en texto plano.
    
    :param abstract_inverted_index: Diccionario con palabras como claves y posiciones como valores.
    :return: Texto plano reconstruido a partir del índice invertido.
    """
    if not abstract_inverted_index:
        return ""
    words = []
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            words.append((position, word))
    words.sort()  # Ordenar por posición
    return " ".join(word for _, word in words)

def search_papers(keyword, per_page=10, max_results=20):
    """
    Busca artículos en la API de OpenAlex utilizando una palabra clave.
    
    :param keyword: Palabra clave para buscar en los títulos de los artículos.
    :param per_page: Número de resultados por página (por defecto, 10).
    :param max_results: Número máximo de resultados a obtener (por defecto, 50).
    :return: Lista de diccionarios con información de los artículos.
    """
    base_url = "https://api.openalex.org/works"
    papers = []
    page = 1  # Página inicial

    while len(papers) < max_results:
        params = {
            "filter": f"title.search:{keyword}",
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Lanza un error si el código de estado no es 200
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error al realizar la solicitud a la API: {e}")
        
        try:
            data = response.json()
        except ValueError:
            raise Exception("Error al decodificar la respuesta JSON de la API.")
        
        results = data.get('results', [])
        if not results:  # Si no hay más resultados, salir del bucle
            break
        
        for item in results:
            # Validar que los campos necesarios existan
            if not item.get("title") or not item.get("abstract_inverted_index"):
                continue
            paper = {
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "abstract": item.get("abstract_inverted_index"),
                "link": item.get("id")
            }
            papers.append(paper)
            if len(papers) >= max_results:  # Detener si alcanzamos el límite
                break
        
        page += 1  # Pasar a la siguiente página
    
    return papers