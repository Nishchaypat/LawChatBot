# Premature and Brute Approach to RAG Application for a Law Article 

# Specs:

The longest text in the content column is 39390, which may not be efficiently embed due to token limitation.

This research presents the development of a legal chatbot designed to assist users in navigating contradicting (similar) legal content with improved precision and speed. The chatbot follows a structured five-stage process:

1. **Data Extraction**
2. **Data Cleaning and Organizing**
3. **Vector Database Conversion**
4. **FAISS and Retrieval-Augmented Generation (RAG)**
5. **Displaying Results**

---

## 1. Data Extraction

The first stage involves extracting legal content from **Title 18, Part 1 Crimes** as a prototype chapter.  
Tools used: **Selenium** and **BeautifulSoup** for automated content retrieval.  

### Key Challenges:
- Variations in formatting and inconsistent titles made fully automated extraction difficult.
- Some sections adhered to standard numbering conventions, while others contained outliers disrupting automation.

**Solution**: Manual intervention was necessary. Data was compiled into **Excel** for refinement.

---

## 2. Data Cleaning and Organizing

In this stage, data structure and quality were enhanced using **VBA macros** in Excel and **NLP techniques** in Python.

### Steps:
- **URL Separation**: Split into individual columns.
- **Keyword Removal**: Unnecessary keywords and empty rows were removed.
- **Metadata Isolation**: Metadata (e.g., citations, references) was moved to a separate column for improved accuracy.

**Benefits**:
- Focus on core legal text for embeddings.
- Metadata retained to provide additional context without compromising precision.

### Output:
- Data organized into three primary columns: **Section Number**, **URL**, and **Content**.
- All 123 chapters of Title 18 were cleaned and stored as separate **CSV files**.

---

## 3. Converting Data to a Vector Database

Using **LegalBERT**, a specialized language model for legal text, the cleaned content was converted into contextual embeddings.

### Key Techniques:
- **HuggingFace Transformers**: AutoTokenizer and AutoModel were used.
- **Performance Optimization**: Model deployed to GPU when available.
- **Embedding Generation**: Averaged token embeddings to create sentence-level representations.

---

## 4. Storing Embeddings in a Cloud-Based Vector Database

**Qdrant**, a cloud-based vector database, was used to store embeddings for similarity search.

### Configuration:
- **Distance Metrics**: Euclidean distance for similarity retrieval.
- **Indexing**: Embeddings paired with metadata (Section and URL).

### Query Handling:
1. Input text passed through `get_embedding` function.
2. Generated embedding queried using **Qdrant's search function**.
3. Results ranked based on smallest Euclidean distance.

---

## 5. User Interface and Result Display

An interactive UI allows users to:
- Enter text excerpts or queries.
- Retrieve section numbers and ranked similar sections.
- Access additional details, including URLs.

**Advantages**:
- Efficient legal research.
- Contextualized retrieval supports decision-making processes.

---

## Conclusion

By integrating web scraping, NLP-based data cleaning, LegalBERT embeddings, and vector search technology, this chatbot bridges the gap between complex legal texts and user-friendly query-based retrieval. It provides a robust tool for modern legal research, offering precision, accessibility, and enhanced decision-making.

---

**This approach was not an intended method for the research, but a stepping stone to an Advanced Agentic RAG application, projected to be made by March 2025.**




Embedding Models tried:

1. nlpaueb/legal-bert-base-uncased
2. Gemini: Text-embedding-005
3. voyage-law-2

Context length:

1. Title18_processed_chapters: 40000 (avg: 16174)
2. LCSC_chapters: 48963 (avg: 1913)
3. RCTS_chapters: 999 (avg: 749)
4. semchunk_chapters: 5567 (avg: 2816)

---

6. Title18_processed_pages: 5867 (avg: 3600)
7. LCSC_pages: 45980 (avg: 1521)
8. RCTS_pages: 1000 (avg: 879)
9. semchunk_pages: 5059 (avg: 2692)

---

1. Average response time,6.36
2. Average query length,636.83
3. Average response length,1533.98
4. Average gemini weight,0.2936
5. Average voyage weight,0.7064

---


   
