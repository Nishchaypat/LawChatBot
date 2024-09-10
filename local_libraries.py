import argparse
from pdfminer.high_level import extract_text
from text_generation import Client
import sentence_transformers
from sentence_transformers import SentenceTransformer, CrossEncoder, util
