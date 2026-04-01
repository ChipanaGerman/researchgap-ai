# ResearchGap AI 🔍

**ResearchGap AI** is an advanced tool designed to help researchers, students, and innovators identify underexplored areas in academic research. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), the application analyzes thousands of research papers to uncover gaps in the literature, providing actionable insights for theses, dissertations, or innovative projects.

---

## 🌐 Demo
[Access the Streamlit App Here](#) *(https://researchgap-ai-eeqjydxt8i8rtkmjexzc5a.streamlit.app/)*

---

## 🧩 The Problem
The overwhelming volume of scientific literature makes it increasingly difficult to identify research niches or underexplored areas. Researchers often spend significant time manually reviewing papers, which slows down innovation and decision-making.

---

## 💡 The Solution
**ResearchGap AI** automates the process of analyzing research papers, clustering them into thematic groups, and detecting gaps in the literature. By combining state-of-the-art NLP techniques (Sentence-Transformers) and clustering algorithms (KMeans), the tool provides a clear and interactive knowledge map to guide users toward areas with research opportunities.

---

## 🔬 Technical Pipeline

1. **Extraction**:
   - Papers are fetched from the **OpenAlex API** based on the user-defined research topic.

2. **NLP Processing**:
   - Abstracts are cleaned and lemmatized using **spaCy**.

3. **Embeddings**:
   - Text embeddings are generated using **all-MiniLM-L6-v2** from HuggingFace.

4. **Clustering**:
   - Papers are grouped into thematic clusters using **Scikit-Learn KMeans**.

5. **Analytics**:
   - **TF-IDF** is used to extract keywords for each cluster.
   - Research gaps are identified based on the number of papers in each cluster.

6. **Visualization**:
   - An interactive 2D knowledge map is created using **UMAP** and **Plotly**.

---

## 🚀 Features

- **Automated Research Analysis**:
  - Fetches and processes papers based on a user-defined topic.
  - Cleans and normalizes abstracts for consistent analysis.

- **Keyword Extraction**:
  - Identifies the most relevant keywords for each cluster using TF-IDF.

- **Clustering and Topic Modeling**:
  - Groups papers into thematic clusters and extracts main topics.

- **Research Gap Detection**:
  - Categorizes clusters as:
    - **Saturated**: Well-researched areas.
    - **Opportunity (Gap)**: Areas with moderate research activity.
    - **Critical Gap**: Areas with very little research.

- **Interactive Knowledge Map**:
  - Visualizes the research landscape in 2D for easy exploration.

---

## 🛠️ Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ChipanaGerman/researchgap-ai.git
   cd researchgap-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Use Case

**Input:**
- Research Topic: Artificial Intelligence in Mental Health  
- Number of Papers: 50  
- Number of Clusters: 5  

**Output:**
- A detailed analysis of research gaps, including cluster summaries and their status.
- An interactive 2D knowledge map showing thematic clusters.

---

## 📚 Dependencies

The project uses the following libraries:

- **Core Libraries**: numpy, pandas, scikit-learn  
- **NLP**: spaCy, sentence-transformers  
- **Visualization**: plotly, umap-learn, streamlit  
- **API Integration**: requests  

For a full list of dependencies, see `requirements.txt`.

---

## 🧠 How It Works

### User Input
The user specifies:
- A research topic  
- The number of papers to analyze  
- The desired number of clusters  

### Processing
- The application fetches papers, cleans the abstracts, and generates embeddings.
- Papers are clustered, and topics are extracted for each cluster.

### Output
- A detailed analysis of research gaps, including cluster summaries and their status.
- An interactive 2D knowledge map for visual exploration of the research landscape.

---

## 🙌 Acknowledgments

- OpenAlex API: For providing access to a vast database of research papers  
- HuggingFace: For the Sentence-Transformers library  
- spaCy: For efficient text processing  

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **Email**: gerluchipanajeronimo@gmail.com 
- **GitHub**: [ChipanaGerman](https://github.com/ChipanaGerman)