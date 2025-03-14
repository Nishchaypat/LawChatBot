import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, SearchRequest
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformerimport
import faiss
import os
import json
from tqdm import tqdm
import dotenv
from datetime import datetime
