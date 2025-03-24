from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd
import os
import numpy as np
import time
from fusionrag import evaluation

load_dotenv()

def load_random_samples() -> List[str]:
    # Load data from the Parquet files
    sections_df = pd.read_parquet("/Users/npatel237/LawChatBot/New_Embeddings_2025/sections/embeddings_gemini_text-005.parquet")
    pages_df = pd.read_parquet("/Users/npatel237/LawChatBot/New_Embeddings_2025/pages/embeddings_gemini_text-005_pages_semchunk.parquet")
    chapters_df = pd.read_parquet("/Users/npatel237/LawChatBot/New_Embeddings_2025/chapters/embeddings_gemini_text-005_chapters_semchunk.parquet")
    
    text_samples = []
    # Extract text based on expected column names
    if "Processed_Content" in sections_df.columns:
        text_samples.extend(sections_df["Processed_Content"].dropna().tolist())
    if "chunk" in pages_df.columns:
        text_samples.extend(pages_df["chunk"].dropna().tolist())
    if "chunk" in chapters_df.columns:
        text_samples.extend(chapters_df["chunk"].dropna().tolist())
    
    print(f"Total samples available: {len(text_samples)}")
    if len(text_samples) < 500:
        print("Warning: Less than 500 samples available, using all available samples.")
        return text_samples
    else:
        return list(np.random.choice(text_samples, size=500, replace=False))

# Define the Pydantic input schema
class QueryInput(BaseModel):
    text: str = Field(..., description="Input text from which queries should be generated")

# Define output schema for structured response
response_schemas = [
    ResponseSchema(name="Expert_Legal", description="Sophisticated legal inquiry with precise terminology, multiple citations, complex structure, and advanced legal discourse"),
    ResponseSchema(name="Professional_Legal", description="Professional legal query with standard terminology, limited citations, clear framing, and practical legal perspective"),
    ResponseSchema(name="Informed_Layperson", description="Query reflecting basic legal knowledge, essential terminology with explanations, and focus on practical implications"),
    ResponseSchema(name="General_Public", description="Non-technical query using everyday language, practical framing, and focus on outcomes and guidance"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define prompt template with escaped curly braces for the JSON block
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Given the following text: "{text}", generate four types of real life user queries that demonstrate different levels of legal expertise and language complexity. Each query should address the same fundamental legal question but be expressed appropriately for different audiences:

1. Expert Legal: Construct a sophisticated legal inquiry as would be drafted by a seasoned attorney or legal scholar. Include:
   - Precise legal terminology and Latin legal maxims where contextually appropriate
   - Multiple specific statutory and case law citations following proper Bluebook format
   - Reference to relevant legal doctrines, standards of review, and judicial tests
   - Complex yet precise syntax reflecting advanced legal discourse
   - Analytical framing demonstrating sophisticated legal knowledge
   - Appropriate jurisdictional context and procedural considerations

2. Professional Legal: Develop a query as would be written by a practicing legal professional. Include:
   - Standard legal terminology without excessive technical language
   - Limited citation to relevant authorities or precedent
   - Clear framing of legal issues with proper context
   - Professional language suitable for legal correspondence
   - Logical structure following accepted legal reasoning patterns
   - Practical legal perspective focused on application

3. Informed Layperson: Create a query reflecting someone with basic legal knowledge but no formal legal training. Include:
   - Fundamental legal concepts with essential terminology
   - Explanations of legal terms when used
   - No formal citations but general references to relevant laws or rights
   - Straightforward structure focusing on understanding
   - Questions framed from a practical problem-solving perspective
   - Focus on implications and real-world applications

4. General Public: Formulate a query using entirely non-technical language accessible to anyone. Include:
   - Everyday vocabulary with no specialized legal terminology
   - Practical, situation-based description of the issue
   - Personal or hypothetical framing (e.g., "What happens if...")
   - Simple, direct sentence structure
   - Focus on outcomes and practical guidance
   - Conversational tone reflecting genuine public inquiry

Respond in valid JSON format using these exact keys:
{{
  "Expert_Legal": "your expert legal query here",
  "Professional_Legal": "your professional legal query here",
  "Informed_Layperson": "your informed layperson query here",
  "General_Public": "your general public query here"
}}
"""
)

# Initialize the AI model (ensure your API key is set and the model exists)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Create the chain using the new composition style
query_chain = prompt_template | llm | output_parser

def generate_queries(text: str):
    input_data = QueryInput(text=text)
    # Use model_dump() to serialize the Pydantic model
    return query_chain.invoke(input_data.model_dump())

if __name__ == "__main__":
    samples = load_random_samples()
    results = []
    for i, sample_text in enumerate(samples):
        try:
            print(f"Processing sample {i+1}/{len(samples)}")
            response = generate_queries(sample_text)
            # Expecting response to be a dict with keys: Expert_Legal, Professional_Legal, Informed_Layperson, General_Public
            results.append(response)
            # Sleep briefly to avoid hitting rate limits (adjust as needed)
            time.sleep(1)
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            # Append a row with None values in case of an error
            results.append({
                "Expert_Legal": None, 
                "Professional_Legal": None, 
                "Informed_Layperson": None, 
                "General_Public": None
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("queries_output.csv", index=False)

    queries = []
    for result in results:
        queries.append(result.get("Expert_Legal"))
        queries.append(result.get("Professional_Legal"))
        queries.append(result.get("Informed_Layperson"))
        queries.append(result.get("General_Public"))
        

