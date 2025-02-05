# Premature and Brute Approach to RAG Application for a Law Article

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

