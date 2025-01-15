import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Carregar o JSON com as regras WCAG
with open("data/wcag_rules.json", "r", encoding="utf-8") as file:
    wcag_rules = json.load(file)

# Ajustar para pegar a chave correta
descricoes = [regra["description"] + " - " + regra["details"] for regra in wcag_rules]

# Inicializar o modelo SBERT
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Gerar embeddings para as descrições
embeddings = modelo.encode(descricoes)

# Salvar embeddings em um arquivo .npz
np.savez("data/wcag_embeddings.npz", embeddings=embeddings)

print("Embeddings gerados e salvos com sucesso!")