import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai
import langchain
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import voyageai
import math
from tqdm import tqdm
import pyarrow.parquet as pq
import spacy
from torch.utils.data import DataLoader, Dataset
from vertexai.language_models import TextEmbeddingModel
from torch import nn
import google.generativeai as genai
import json
from typing import List, Dict, Union, Optional, Any
from fuzzy import LegalQueryAnalyzer


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\mkolla1\LawChatBot\gcpservicekey.json"
except:
    print("Error at Try block")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"gcpservicekey.json"
    
PROJECT_ID = "lawrag"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


voyageai.api_key = os.getenv("VOYAGE_API")


'''
Embedding Generator class to generate embeddings using Gemini and VoyageAI models.
'''

class EmbeddingGenerator:
    def __init__(self, gemini_model_name="text-embedding-005", voyage_model_name="voyage-law-2"):
        """
        Initializes the embedding generator with Gemini and VoyageAI models.
        """
        self.gemini_model = VertexAIEmbeddings(gemini_model_name)
        self.voyage_model_name = voyage_model_name
        self.voyage_client = voyageai.Client()
        self.voyage_tockenizer=AutoTokenizer.from_pretrained('voyageai/voyage-2')
        self.gemini_tockenizer=tiktoken.get_encoding("cl100k_base")

    def chunk_text_gemini(self, text, max_tokens=4096, overlap=512):
        tokens = self.gemini_tockenizer.encode(text)

        chunks = []
        start = 0
        while start < len(tokens):
            chunk = tokens[start:start + max_tokens]
            chunks.append(self.gemini_tockenizer.decode(chunk))
            start += max_tokens - overlap  # Sliding window
        return chunks
    
    def chunk_text_voyage(self, text, max_tokens=4096, overlap=512):
        """
        Splits text into chunks based on the token limit of voyage-law-2 tokenizer.
        Uses a sliding window approach with overlap.
        
        Args:
            text (str): The input text to be chunked.
            max_tokens (int): Maximum tokens per chunk (4096 for voyage-law-2).
            overlap (int): Overlapping tokens to maintain context between chunks.

        Returns:
            list of str: List of text chunks.
        """
        tokens = self.voyage_tockenizer.encode(text, add_special_tokens=False)

        chunks = []
        start = 0
        while start < len(tokens):
            chunk = tokens[start:start + max_tokens]
            chunks.append(self.voyage_tockenizer.decode(chunk))
            start += max_tokens - overlap

        return chunks
    
    def get_embeddings_gemini(self, texts, batch_size=32):
        """
        Compute embeddings using VertexAIEmbeddings in batches.

        Args:
            texts (list of str): List of text data to embed.
            batch_size (int): Number of texts to process per batch.

        Returns:
            list: List of embedding vectors.
        """
        embeddings = []
        texts= self.chunk_text_gemini(texts[0])
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i:i + batch_size]  # Get batch of texts
            batch_embeddings = self.gemini_model.embed_documents(batch)  # Generate embeddings
            embeddings.extend(batch_embeddings)  # Store results

        return embeddings

    def get_embeddings_voyage(self, texts, batch_size=32):
        """
        Compute embeddings using the VoyageAI Python client in batches.

        Args:
            texts (list of str): List of text data to embed.
            batch_size (int): Number of texts per batch.

        Returns:
            list: List of embedding vectors.
        """
        embeddings = []
        texts= self.chunk_text_voyage(texts[0])
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size] 
            
            try:
                response = self.voyage_client.embed(batch, model=self.voyage_model_name)
                batch_embeddings = response.embeddings  
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")

        return embeddings


'''
Legal Embedding Loader class to load embeddings from parquet files for both models and all granularity levels.
'''


class LegalEmbeddingLoader:
    """Loads embeddings from parquet files for both models and all granularity levels."""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.gemini_embeddings = {}
        self.voyager_embeddings = {}
        self.metadata = {}
        
    def load_embeddings(self):
        """Load the six specified embedding files."""
        file_mappings = {
            "gemini_chapters": "embeddings_gemini_text-005_chapters_semchunk.parquet",
            "voyager_chapters": "embeddings_voyage_per_chapter_semchunked.parquet",
            "gemini_pages": "embeddings_gemini_text-005_pages_semchunk.parquet",
            "voyager_pages": "embeddings_voyage_per_pages_semchunked.parquet",
            "gemini_sections": "embeddings_gemini_text-005.parquet",
            "voyager_sections": "embeddings_voyage.parquet",
        }

        for key, file_name in file_mappings.items():
            print(self.base_path)
            file_path = os.path.join(self.base_path, key.split("_")[-1], file_name)
            print(file_path)
            if not os.path.exists(file_path):
                print(f"File {file_name} not found. Skipping...")
                continue

            # Read parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            print(f"\nColumns in {file_name}: {df.columns.tolist()}")
            # Extract embeddings and metadata
            embeddings = np.stack(df["Embedding"].values)

            # Determine model and granularity
            model, granularity = key.split("_")

            # Store embeddings
            if model == "gemini":
                self.gemini_embeddings[granularity] = torch.tensor(embeddings, dtype=torch.float32)
            else:
                self.voyager_embeddings[granularity] = torch.tensor(embeddings, dtype=torch.float32)

            # Store metadata
            self.metadata[key] = df.drop('Embedding', axis=1)

            print(f"Loaded {file_name} ({model} - {granularity})")
        return self.gemini_embeddings, self.voyager_embeddings, self.metadata

    def get_embedding_dimensions(self):
        """Return the dimensions of embeddings for both models."""
        gemini_dim = {k: v.shape[1] for k, v in self.gemini_embeddings.items()}
        voyager_dim = {k: v.shape[1] for k, v in self.voyager_embeddings.items()}
        return gemini_dim, voyager_dim


'''
FusionRetrival class to perform document retrieval using the fusion of Gemini and VoyageAI embeddings.
''' 

class FusionRetrival(nn.Module):
    def __init__(self, output_dim=1024, top_n=10):
        super(FusionRetrival, self).__init__()
        
        self.output_dim = output_dim
        self.top_n = top_n
        self.granularities = ['sections', 'chapters', 'pages']

        # Query projectors
        self.gemini_query_projector = nn.Linear(768, 768)
        self.voyager_query_projector = nn.Linear(1024, output_dim)
        
        # Document projectors
        self.gemini_projector = nn.Linear(768, 768)
        self.voyager_projector = nn.Linear(1024, output_dim)

        # Aggregation
        self.layer_norm = nn.LayerNorm(output_dim)


    def forward(self, gemini_embeddings, voyager_embeddings, gemini_query_embedding, voyager_query_embedding):
        # Compute cosine similarities directly without projections and normalizations
        gemini_section_similarities = F.cosine_similarity(gemini_embeddings['sections'], gemini_query_embedding)
        gemini_chapter_similarities = F.cosine_similarity(gemini_embeddings['chapters'], gemini_query_embedding)
        gemini_page_similarities = F.cosine_similarity(gemini_embeddings['pages'], gemini_query_embedding)

        voyager_section_similarities = F.cosine_similarity(voyager_embeddings['sections'], voyager_query_embedding)
        voyager_chapter_similarities = F.cosine_similarity(voyager_embeddings['chapters'], voyager_query_embedding)
        voyager_page_similarities = F.cosine_similarity(voyager_embeddings['pages'], voyager_query_embedding)

        # Apply softmax to the similarity scores
        gemini_section_weights = F.softmax(gemini_section_similarities, dim=0)
        gemini_chapter_weights = F.softmax(gemini_chapter_similarities, dim=0)
        gemini_page_weights = F.softmax(gemini_page_similarities, dim=0)

        voyager_section_weights = F.softmax(voyager_section_similarities, dim=0)
        voyager_chapter_weights = F.softmax(voyager_chapter_similarities, dim=0)
        voyager_page_weights = F.softmax(voyager_page_similarities, dim=0)

        print(f"Shape of gemini_section_weights: {gemini_section_weights.shape}")
        print(f"Shape of voyager_section_weights: {voyager_section_weights.shape}")
        print(f"Shape of gemini_chapter_weights: {gemini_chapter_weights.shape}")
        print(f"Shape of voyager_chapter_weights: {voyager_chapter_weights.shape}")
        print(f"Shape of gemini_page_weights: {gemini_page_weights.shape}")
        print(f"Shape of voyager_page_weights: {voyager_page_weights.shape}")
        # Get top N indices
        gemini_section_top_values, gemini_section_top_indices = torch.topk(gemini_section_weights, self.top_n)
        gemini_chapter_top_values, gemini_chapter_top_indices = torch.topk(gemini_chapter_weights, self.top_n)
        gemini_page_top_values, gemini_page_top_indices = torch.topk(gemini_page_weights, self.top_n)

        voyager_section_top_values, voyager_section_top_indices = torch.topk(voyager_section_weights, self.top_n)
        voyager_chapter_top_values, voyager_chapter_top_indices = torch.topk(voyager_chapter_weights, self.top_n)
        voyager_page_top_values, voyager_page_top_indices = torch.topk(voyager_page_weights, self.top_n)

        # Store results in dictionary
        top_indices_dict = {
            'sections': {
                "gemini_top_values": gemini_section_top_values,
                "gemini_top_indices": gemini_section_top_indices,
                "voyager_top_values": voyager_section_top_values,
                "voyager_top_indices": voyager_section_top_indices
            },
            'chapters': {
                "gemini_top_values": gemini_chapter_top_values,
                "gemini_top_indices": gemini_chapter_top_indices,
                "voyager_top_values": voyager_chapter_top_values,
                "voyager_top_indices": voyager_chapter_top_indices
            },
            'pages': {
                "gemini_top_values": gemini_page_top_values,
                "gemini_top_indices": gemini_page_top_indices,
                "voyager_top_values": voyager_page_top_values,
                "voyager_top_indices": voyager_page_top_indices
            }
        }

        return {
            'top_indices': top_indices_dict
        }

    def forward2(self, gemini_embeddings, voyager_embeddings, gemini_query_embedding, voyager_query_embedding):
        # Project query embeddings
        gemini_query_embedding = gemini_query_embedding.unsqueeze(0)
        voyager_query_embedding = voyager_query_embedding.unsqueeze(0)
        print(f"gemini_query_embedding: {gemini_query_embedding.shape}")
        print(f"voyager_query_embedding: {voyager_query_embedding.shape}")
        gemini_query_projected = self.gemini_query_projector(gemini_query_embedding)
        voyager_query_projected = self.voyager_query_projector(voyager_query_embedding)

        # Normalize query embeddings
        gemini_query_norm = F.normalize(gemini_query_projected, p=2, dim=1)
        voyager_query_norm = F.normalize(voyager_query_projected, p=2, dim=1)
        
        # Process embeddings per granularity
        gemini_section_proj = self.gemini_projector(gemini_embeddings['sections'])
        gemini_chapter_proj = self.gemini_projector(gemini_embeddings['chapters'])
        gemini_page_proj = self.gemini_projector(gemini_embeddings['pages'])

        voyager_section_proj = self.voyager_projector(voyager_embeddings['sections'])
        voyager_chapter_proj = self.voyager_projector(voyager_embeddings['chapters'])
        voyager_page_proj = self.voyager_projector(voyager_embeddings['pages'])

        gemini_section_proj = gemini_embeddings['sections']
        gemini_chapter_proj = gemini_embeddings['chapters']
        gemini_page_proj = gemini_embeddings['pages']

        voyager_section_proj = voyager_embeddings['sections']
        voyager_chapter_proj = voyager_embeddings['chapters']
        voyager_page_proj = voyager_embeddings['pages']

        # Normalize document embeddings
        gemini_section_norm = F.normalize(gemini_section_proj, p=2, dim=1)
        gemini_chapter_norm = F.normalize(gemini_chapter_proj, p=2, dim=1)
        gemini_page_norm = F.normalize(gemini_page_proj, p=2, dim=1)

        voyager_section_norm = F.normalize(voyager_section_proj, p=2, dim=1)
        voyager_chapter_norm = F.normalize(voyager_chapter_proj, p=2, dim=1)
        voyager_page_norm = F.normalize(voyager_page_proj, p=2, dim=1)
        

        # Compute cosine similarity using matrix multiplication
        gemini_section_similarities = torch.matmul(gemini_section_norm, gemini_query_norm.T).squeeze(1)
        gemini_chapter_similarities = torch.matmul(gemini_chapter_norm, gemini_query_norm.T).squeeze(1)
        gemini_page_similarities = torch.matmul(gemini_page_norm, gemini_query_norm.T).squeeze(1)

        voyager_section_similarities = torch.matmul(voyager_section_norm, voyager_query_norm.T).squeeze(1)
        voyager_chapter_similarities = torch.matmul(voyager_chapter_norm, voyager_query_norm.T).squeeze(1)
        voyager_page_similarities = torch.matmul(voyager_page_norm, voyager_query_norm.T).squeeze(1)


        # Apply softmax
        gemini_section_weights = F.softmax(gemini_section_similarities, dim=0)
        gemini_chapter_weights = F.softmax(gemini_chapter_similarities, dim=0)
        gemini_page_weights = F.softmax(gemini_page_similarities, dim=0)

        voyager_section_weights = F.softmax(voyager_section_similarities, dim=0)
        voyager_chapter_weights = F.softmax(voyager_chapter_similarities, dim=0)
        voyager_page_weights = F.softmax(voyager_page_similarities, dim=0)

        print(f"Shape of gemini_section_weights: {gemini_section_weights.shape}")
        print(f"Shape of voyager_section_weights: {voyager_section_weights.shape}")
        print(f"Shape of gemini_chapter_weights: {gemini_chapter_weights.shape}")
        print(f"Shape of voyager_chapter_weights: {voyager_chapter_weights.shape}")
        print(f"Shape of gemini_page_weights: {gemini_page_weights.shape}")
        print(f"Shape of voyager_page_weights: {voyager_page_weights.shape}")
        # Get top N indices
        gemini_section_top_values, gemini_section_top_indices = torch.topk(gemini_section_weights, self.top_n)
        gemini_chapter_top_values, gemini_chapter_top_indices = torch.topk(gemini_chapter_weights, self.top_n)
        gemini_page_top_values, gemini_page_top_indices = torch.topk(gemini_page_weights, self.top_n)

        voyager_section_top_values, voyager_section_top_indices = torch.topk(voyager_section_weights, self.top_n)
        voyager_chapter_top_values, voyager_chapter_top_indices = torch.topk(voyager_chapter_weights, self.top_n)
        voyager_page_top_values, voyager_page_top_indices = torch.topk(voyager_page_weights, self.top_n)

        # Store results in dictionary
        top_indices_dict = {
            'sections': {
                "gemini_top_values": gemini_section_top_values,
                "gemini_top_indices": gemini_section_top_indices,
                "voyager_top_values": voyager_section_top_values,
                "voyager_top_indices": voyager_section_top_indices
            },
            'chapters': {
                "gemini_top_values": gemini_chapter_top_values,
                "gemini_top_indices": gemini_chapter_top_indices,
                "voyager_top_values": voyager_chapter_top_values,
                "voyager_top_indices": voyager_chapter_top_indices
            },
            'pages': {
                "gemini_top_values": gemini_page_top_values,
                "gemini_top_indices": gemini_page_top_indices,
                "voyager_top_values": voyager_page_top_values,
                "voyager_top_indices": voyager_page_top_indices
            }
        }
        print("FINAL")
        return {
            'top_indices': top_indices_dict
        }

    def forward1(self, gemini_embeddings, voyager_embeddings, gemini_query_embedding, voyager_query_embedding):
        # Project query embeddings
        gemini_query_embedding = gemini_query_embedding.unsqueeze(0)
        voyager_query_embedding = voyager_query_embedding.unsqueeze(0)
        # print(f"gemini_query_embedding: {gemini_query_embedding.shape}")
        # print(f"voyager_query_embedding: {voyager_query_embedding.shape}")
        # gemini_query_projected = self.gemini_query_projector(gemini_query_embedding)
        # voyager_query_projected = self.voyager_query_projector(voyager_query_embedding)

        # Normalize query embeddings
        gemini_query_norm = F.normalize(gemini_query_embedding, p=2, dim=1)
        voyager_query_norm = F.normalize(voyager_query_embedding, p=2, dim=1)
        
        # Process embeddings per granularity
        # gemini_section_proj = self.gemini_projector(gemini_embeddings['sections'])
        # gemini_chapter_proj = self.gemini_projector(gemini_embeddings['chapters'])
        # gemini_page_proj = self.gemini_projector(gemini_embeddings['pages'])

        # voyager_section_proj = self.voyager_projector(voyager_embeddings['sections'])
        # voyager_chapter_proj = self.voyager_projector(voyager_embeddings['chapters'])
        # voyager_page_proj = self.voyager_projector(voyager_embeddings['pages'])

        gemini_section_proj = gemini_embeddings['sections']
        gemini_chapter_proj = gemini_embeddings['chapters']
        gemini_page_proj = gemini_embeddings['pages']

        voyager_section_proj = voyager_embeddings['sections']
        voyager_chapter_proj = voyager_embeddings['chapters']
        voyager_page_proj = voyager_embeddings['pages']

        # Normalize document embeddings
        gemini_section_norm = F.normalize(gemini_section_proj, p=2, dim=1)
        gemini_chapter_norm = F.normalize(gemini_chapter_proj, p=2, dim=1)
        gemini_page_norm = F.normalize(gemini_page_proj, p=2, dim=1)

        voyager_section_norm = F.normalize(voyager_section_proj, p=2, dim=1)
        voyager_chapter_norm = F.normalize(voyager_chapter_proj, p=2, dim=1)
        voyager_page_norm = F.normalize(voyager_page_proj, p=2, dim=1)
        

        # Compute cosine similarity using matrix multiplication
        gemini_section_similarities = torch.matmul(gemini_section_norm, gemini_query_norm.T).squeeze(1)
        gemini_chapter_similarities = torch.matmul(gemini_chapter_norm, gemini_query_norm.T).squeeze(1)
        gemini_page_similarities = torch.matmul(gemini_page_norm, gemini_query_norm.T).squeeze(1)

        voyager_section_similarities = torch.matmul(voyager_section_norm, voyager_query_norm.T).squeeze(1)
        voyager_chapter_similarities = torch.matmul(voyager_chapter_norm, voyager_query_norm.T).squeeze(1)
        voyager_page_similarities = torch.matmul(voyager_page_norm, voyager_query_norm.T).squeeze(1)


        # Apply softmax
        gemini_section_weights = F.softmax(gemini_section_similarities, dim=0)
        gemini_chapter_weights = F.softmax(gemini_chapter_similarities, dim=0)
        gemini_page_weights = F.softmax(gemini_page_similarities, dim=0)

        voyager_section_weights = F.softmax(voyager_section_similarities, dim=0)
        voyager_chapter_weights = F.softmax(voyager_chapter_similarities, dim=0)
        voyager_page_weights = F.softmax(voyager_page_similarities, dim=0)

        print(f"Shape of gemini_section_weights: {gemini_section_weights.shape}")
        print(f"Shape of voyager_section_weights: {voyager_section_weights.shape}")
        print(f"Shape of gemini_chapter_weights: {gemini_chapter_weights.shape}")
        print(f"Shape of voyager_chapter_weights: {voyager_chapter_weights.shape}")
        print(f"Shape of gemini_page_weights: {gemini_page_weights.shape}")
        print(f"Shape of voyager_page_weights: {voyager_page_weights.shape}")
        # Get top N indices
        gemini_section_top_values, gemini_section_top_indices = torch.topk(gemini_section_weights, self.top_n)
        gemini_chapter_top_values, gemini_chapter_top_indices = torch.topk(gemini_chapter_weights, self.top_n)
        gemini_page_top_values, gemini_page_top_indices = torch.topk(gemini_page_weights, self.top_n)

        voyager_section_top_values, voyager_section_top_indices = torch.topk(voyager_section_weights, self.top_n)
        voyager_chapter_top_values, voyager_chapter_top_indices = torch.topk(voyager_chapter_weights, self.top_n)
        voyager_page_top_values, voyager_page_top_indices = torch.topk(voyager_page_weights, self.top_n)

        # Store results in dictionary
        top_indices_dict = {
            'sections': {
                "gemini_top_values": gemini_section_top_values,
                "gemini_top_indices": gemini_section_top_indices,
                "voyager_top_values": voyager_section_top_values,
                "voyager_top_indices": voyager_section_top_indices
            },
            'chapters': {
                "gemini_top_values": gemini_chapter_top_values,
                "gemini_top_indices": gemini_chapter_top_indices,
                "voyager_top_values": voyager_chapter_top_values,
                "voyager_top_indices": voyager_chapter_top_indices
            },
            'pages': {
                "gemini_top_values": gemini_page_top_values,
                "gemini_top_indices": gemini_page_top_indices,
                "voyager_top_values": voyager_page_top_values,
                "voyager_top_indices": voyager_page_top_indices
            }
        }
        print("FINAL")
        return {
            'top_indices': top_indices_dict
        }




def remove_duplicates(top_indices_dict):
    """
    Removes duplicate indices within each granularity (sections, chapters, pages)
    while keeping the one with the higher value. If values are equal, remove from Gemini.
    """
    for granularity in top_indices_dict:
        gemini_indices = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in top_indices_dict[granularity]["gemini_top_indices"]]
        gemini_values = [float(val.item()) if hasattr(val, 'item') else float(val) for val in top_indices_dict[granularity]["gemini_top_values"]]
        voyager_indices = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in top_indices_dict[granularity]["voyager_top_indices"]]
        voyager_values = [float(val.item()) if hasattr(val, 'item') else float(val) for val in top_indices_dict[granularity]["voyager_top_values"]]
        
        gemini_map = {idx: val for idx, val in zip(gemini_indices, gemini_values)}
        voyager_map = {idx: val for idx, val in zip(voyager_indices, voyager_values)}
        
        # Identify duplicates and remove based on the given rule
        common_indices = set(gemini_map.keys()).intersection(set(voyager_map.keys()))
        for idx in common_indices:
            if gemini_map[idx] > voyager_map[idx]:
                del voyager_map[idx]
            else:
                del gemini_map[idx]
        
        # Update lists after processing
        top_indices_dict[granularity]["gemini_top_indices"] = list(gemini_map.keys())
        top_indices_dict[granularity]["gemini_top_values"] = list(gemini_map.values())
        top_indices_dict[granularity]["voyager_top_indices"] = list(voyager_map.keys())
        top_indices_dict[granularity]["voyager_top_values"] = list(voyager_map.values())
    
    return top_indices_dict

def get_weighted_indices(top_indices_dict, gemini_weight, voyager_weight):
    """
    Selects a weighted number of indices based on provided weights.
    """
    for granularity in top_indices_dict:
        gemini_indices = top_indices_dict[granularity]["gemini_top_indices"]
        gemini_values = top_indices_dict[granularity]["gemini_top_values"]
        voyager_indices = top_indices_dict[granularity]["voyager_top_indices"]
        voyager_values = top_indices_dict[granularity]["voyager_top_values"]
        
        num_gemini = math.ceil(len(gemini_indices) * gemini_weight)
        num_voyager = math.ceil(len(voyager_indices) * voyager_weight)
        
        top_indices_dict[granularity]["gemini_top_indices"] = gemini_indices[:num_gemini]
        top_indices_dict[granularity]["gemini_top_values"] = gemini_values[:num_gemini]
        top_indices_dict[granularity]["voyager_top_indices"] = voyager_indices[:num_voyager]
        top_indices_dict[granularity]["voyager_top_values"] = voyager_values[:num_voyager]
    
    print("FINAL")
    return {
        'top_indices': top_indices_dict
    }




'''

Complete RAG class to perform document retrieval using the fusion of Gemini and VoyageAI embeddings.
'''



def testing(base_path, query, forward_fn, filter_fn):
    """
    Test function for MultiLevelAttention model using real embeddings from saved files.
    
    Args:
        base_path (str): Path to the directory containing embedding parquet files.
        query_embedding (torch.Tensor): Query embedding tensor with shape [768].
    
    Returns:
        None
    """
    # Load embeddings
    loader = LegalEmbeddingLoader(base_path)
    gemini_embeddings, voyager_embeddings, metadata = loader.load_embeddings()

    # Initialize the model
    model = FusionRetrival(output_dim=1024)

    # Analyze query
    analyzer = LegalQueryAnalyzer(use_gemini=True)

    query_weights = analyzer.analyze_query(query)
    print("&&&&&&&&&&&&&&&&&")
    print(query_weights['weights']['gemini'])
    print(query_weights['weights']['voyager'])

    embedder = EmbeddingGenerator()
    gemini_query_embeddings= embedder.get_embeddings_gemini([query])
    voyage_query_embeddings= embedder.get_embeddings_voyage([query])
    query_output = {'gemini_embedding': gemini_query_embeddings[0], 'voyage_embedding': voyage_query_embeddings[0]}
    print("********************************")
    # Run the model
    if forward_fn == "simple":
        output = model.forward(
            gemini_embeddings, 
            voyager_embeddings, 
            torch.tensor(query_output['voyage_embedding']),
            torch.tensor(query_output['voyage_embedding'])
        )
    else:
        output = model.forward1(
            gemini_embeddings, 
            voyager_embeddings, 
            torch.tensor(query_output['gemini_embedding']),
            torch.tensor(query_output['voyage_embedding'])
        )
    
    for granularity, data in output['top_indices'].items():
        print(f"\n--- {granularity.capitalize()} ---")
        print(f"Gemini Top Indices: {data['gemini_top_indices']} GEMINI TOP VALUES: {data['gemini_top_values']}")
        print(f"Voyager Top Indices: {data['voyager_top_indices']} VOYAGER TOP VALUES: {data['voyager_top_values']}")
    # print("AFTER Processing and WEIGHTED")

    

    if filter_fn:
        cleaned_top_indices = remove_duplicates(output['top_indices'])
        final_indices = get_weighted_indices(cleaned_top_indices, query_weights['weights']['gemini'], query_weights['weights']['voyager'])
        print(final_indices)
        for granularity, data in final_indices['top_indices'].items():
            print(f"\n--- {granularity.capitalize()} ---")
            print(f"Gemini Top Indices: {data['gemini_top_indices']} GEMINI TOP VALUES: {data['gemini_top_values']}")
            print(f"Voyager Top Indices: {data['voyager_top_indices']} VOYAGER TOP VALUES: {data['voyager_top_values']}")
        weights = {'gemini_weight': query_weights['weights']['gemini'], 'voyager_weight': query_weights['weights']['voyager']}
        final_output = {**final_indices, **weights}
        return final_output
    else:
        return output
# base_path = "New_Embeddings_2025" 
# query = """Can a vessel be seized and forfeited to the United States if the owner or master knowingly allows it to be used for conspiring against the United States?
# """
# testing(base_path, query, forward_fn="", filter_fn= True)
