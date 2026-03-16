from data_fetcher import search_papers, convert_abstract_inverted_index_to_text

# 1. Buscar papers sobre un tema
tema = "Artificial Intelligence in Mental Health"
print(f"Buscando papers sobre: {tema}...")
papers = search_papers(tema)

# 2. Mostrar resultados y limpiar el abstract
for p in papers:
    abstract_texto = convert_abstract_inverted_index_to_text(p['abstract'])
    print(f"\nTítulo: {p['title']}")
    print(f"Año: {p['year']}")
    print(f"Abstract (primeros 100 caracteres): {abstract_texto[:100]}...")