from fusion import testing
import pandas as pd
import google.generativeai as genai
import os
import time

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Load the embeddings files
gemini_sections_df = pd.read_parquet(r"New_Embeddings_2025\sections\embeddings_gemini_text-005.parquet")
gemini_pages_df = pd.read_parquet(r"New_Embeddings_2025\pages\embeddings_gemini_text-005_pages_semchunk.parquet")
gemini_chapters_df = pd.read_parquet(r"New_Embeddings_2025\chapters\embeddings_gemini_text-005_chapters_semchunk.parquet")

voyage_sections_df = pd.read_parquet(r"New_Embeddings_2025\sections\embeddings_voyage.parquet")
voyage_pages_df = pd.read_parquet(r"New_Embeddings_2025\pages\embeddings_voyage_per_pages_semchunked.parquet")
voyage_chapters_df = pd.read_parquet(r"New_Embeddings_2025\chapters\embeddings_voyage_per_chapter_semchunked.parquet")

'''
Retriving the Content based on the Indices retrived from the Retrival process
'''
def get_processed_content_by_index(index, source, model):
    """
    Retrieve the Processed_Content or chunk based on the specified index and source.

    Args:
        index (int): The row index to retrieve.
        source (str): The source dataset ("gemini_text", "gemini_pages", or "voyage").

    Returns:
        str: The corresponding Processed_Content or chunk.
    """
    if model== "gemini":
        if source == "sections":
            return gemini_sections_df.loc[index, "Processed_Content"] if index in gemini_sections_df.index else None
        elif source == "pages":
            return gemini_pages_df.loc[index, "chunk"] if index in gemini_pages_df.index else None
        elif source == "chapters":
            return gemini_chapters_df.loc[index, "chunk"] if index in gemini_chapters_df.index else None
        else:
            return "Invalid source specified."
    if model == "voyage":
        if source == "sections":
            return voyage_sections_df.loc[index, "Processed_Content"] if index in voyage_sections_df.index else None
        elif source == "pages":
            return voyage_pages_df.loc[index, "chunk"] if index in voyage_pages_df.index else None
        elif source == "chapters":
            return voyage_chapters_df.loc[index, "chunk"] if index in voyage_chapters_df.index else None
        else:
            return "Invalid source specified."



# print(structured_data)
prompt= """You are a highly specialized legal expert and research assistant. Your expertise is strictly confined to legal principles, case law, statutes, and regulatory frameworks. You possess a deep understanding of legal terminology and can accurately interpret complex legal information.

Your primary function is to provide precise and concise answers to legal questions based on the provided context. You will first retrieve relevant information from the provided legal documents and then synthesize a response that directly addresses the user's query.

**Instructions:**

1.  **Contextual Analysis:** Carefully analyze the provided legal documents, including data from various sections, pages, and chapters of the two distinct models, to understand the relevant facts, legal principles, and applicable laws.
2.  **Information Retrieval:** Extract the maximum amount of information directly pertinent to the user's question from all provided sections, pages, and chapters of both models.
3.  **Cross-Model Integration:** If the two models provide differing or complementary information on the same legal issue, integrate these perspectives into a coherent and comprehensive response.
4.  **Synthesis and Response:** Formulate a clear, accurate, and concise answer using the retrieved information. Cite specific legal sources (e.g., case names, statute sections, model names, page numbers) when applicable.
5.  **Clarity and Precision:** Use precise legal terminology and avoid jargon whenever possible. If complex terminology is necessary, provide a brief explanation.
6.  **No Extrapolation:** Do not make assumptions or extrapolate beyond the information contained in the provided context. Only answer based on the context.
7.  **No personal opinions:** Do not provide personal opinions or legal advice. Only provide information based on the law.

** Input Data Type:**
The input data consists of structured information extracted from two distinct models (Gemini and Voyage) that includes sections, pages, and chapters relevant to the legal question in dictionary format. Here is the data {data_here}

**Task:**

Answer the following legal question based on the provided content: {query_here}
"""

'''
Augmented generation with Prompt Engineering
'''
def generate_response(prompt,  query, data):
    model = genai.GenerativeModel('models/gemini-2.0-flash',
                              system_instruction=prompt.format(query_here=query, data_here=data),
                                )
    response = model.generate_content("Please provide a concise and accurate response based on the provided legal documents and query.")
    if response:
        return response.text
    else:
        return "No response generated. Please check the input data or model configuration."


'''
Evaluation Code for Testing Purposes
'''
def evaluation(queries):
    data = []  # List to store the data for the DataFrame
    for query in queries:
        start_time = time.time()  # Record the start time
        
        base_path = "New_Embeddings_2025" 
        results = testing(base_path, query, forward_fn="Advanced", filter_fn=True)
        
        structured_data = {
            "sections": {
                "gemini": list(map(lambda x: get_processed_content_by_index(x, 'sections', 'gemini'), results['top_indices']['sections']['gemini_top_indices'])),
                "voyage": list(map(lambda x: get_processed_content_by_index(x, 'sections', 'voyage'), results['top_indices']['sections']['voyager_top_indices'])),
            },
            "chapters": {
                "gemini": list(map(lambda x: get_processed_content_by_index(x, 'chapters', 'gemini'), results['top_indices']['chapters']['gemini_top_indices'])),
                "voyage": list(map(lambda x: get_processed_content_by_index(x, 'chapters', 'voyage'), results['top_indices']['chapters']['voyager_top_indices'])),
            },
            "pages": {
                "gemini": list(map(lambda x: get_processed_content_by_index(x, 'pages', 'gemini'), results['top_indices']['pages']['gemini_top_indices'])),
                "voyage": list(map(lambda x: get_processed_content_by_index(x, 'pages', 'voyage'), results['top_indices']['pages']['voyager_top_indices'])),
            }
        }
        
        gemini_response = generate_response(prompt, query, structured_data)
        
        end_time = time.time()  # Record the end time
        time_taken = end_time - start_time  # Calculate the time taken for this loop
        
        # Append the data for the current query
        data.append({
            "Query": query,
            "Gemini Weights": results['gemini_weight'],
            "Voyager Weights": results['voyager_weight'],
            "Results": results['top_indices'],
            "Gemini Response": gemini_response,
            "Time Taken (seconds)": time_taken
        })
    
    # Create the DataFrame from the collected data
    df = pd.DataFrame(data)
    df.to_csv(r"Evaluation_Results.csv")




'''
Final Function to retrive the indices, text, and generate response
'''
def LegalChatBot(query):
    base_path = "New_Embeddings_2025" 
    results= testing(base_path, query, forward_fn="Advanced", filter_fn= True)
    print(f"Weights of Gemini: {results['gemini_weight']}")
    print(f"Weights of Voyage: {results['voyager_weight']}")
    structured_data = {
    "sections": {
        "gemini": list(map(lambda x: get_processed_content_by_index(x, 'sections','gemini' ), results['top_indices']['sections']['gemini_top_indices'])),
        "voyage": list(map(lambda x: get_processed_content_by_index(x, 'sections','voyage'), results['top_indices']['sections']['voyager_top_indices'])),
    },
    "chapters": {
        "gemini": list(map(lambda x: get_processed_content_by_index(x, 'chapters', 'gemini'), results['top_indices']['chapters']['gemini_top_indices'])),
        "voyage": list(map(lambda x: get_processed_content_by_index(x, 'chapters', 'voyage'), results['top_indices']['chapters']['voyager_top_indices'])),
    },
    "pages": {
        "gemini": list(map(lambda x: get_processed_content_by_index(x, 'pages', 'gemini'), results['top_indices']['pages']['gemini_top_indices'])),
        "voyage": list(map(lambda x: get_processed_content_by_index(x, 'pages', 'voyage'), results['top_indices']['pages']['voyager_top_indices'])),
    }
}
    print("#################################################################################################")
    print("Gemini Response:")
    gemini_response = generate_response(prompt, query, structured_data)
    print(gemini_response)
    print("#################################################################################################")
    return gemini_response


# test here
query = """Can a crime victim seek advice from an attorney regarding their rights, as described in subsection (a)?"""
LegalChatBot(query)





