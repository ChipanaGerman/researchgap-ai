from data_fetcher import search_papers, convert_abstract_inverted_index_to_text
from processor import TextProcessor

# 1. Buscar papers sobre un tema
tema = "Artificial Intelligence in Mental Health"
print(f"Buscando papers sobre: {tema}...")
papers = search_papers(tema)

# 2. Procesar los abstracts
processor = TextProcessor()
abstracts = [convert_abstract_inverted_index_to_text(p['abstract']) for p in papers]
cleaned_abstracts = processor.clean_texts(abstracts)

# 3. Mostrar resultados limpios y extraer palabras clave
for i, p in enumerate(papers):
    print(f"\nTítulo: {p['title']}")
    print(f"Año: {p['year']}")
    print(f"Abstract limpio: {cleaned_abstracts[i]}")

# 4. Extraer palabras clave principales
top_keywords = processor.extract_top_keywords(cleaned_abstracts, top_n=5)
for i, keywords in enumerate(top_keywords):
    print(f"\nPalabras clave del artículo '{papers[i]['title']}': {', '.join(keywords)}")