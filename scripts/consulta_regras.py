import json
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import generate

# Carrega o JSON com as regras WCAG
with open("data/wcag_rules.json", "r", encoding="utf-8") as file:
    wcag_rules = json.load(file)

# Carrega os embeddings pré-calculados
embeddings_data = np.load("data/wcag_embeddings.npz")
embeddings = embeddings_data["embeddings"]

# Inicializa o modelo SBERT
modelo = SentenceTransformer("all-MiniLM-L6-v2")

def encontra_regra_relevante(pergunta):
    """
    Encontra a regra WCAG mais relevante com base na pergunta do usuário.
    """
    # Gera o embedding para a pergunta
    embedding_pergunta = modelo.encode([pergunta])

    # Calcula a similaridade
    similaridades = np.dot(embeddings, embedding_pergunta.T).squeeze()
    indice_mais_similar = np.argmax(similaridades)

    # Retorna a regra mais relevante
    return wcag_rules[indice_mais_similar]


def consulta_ollama(pergunta, regra_relevante):
    """
    Consulta o modelo Ollama com a pergunta do usuário e a regra relevante.
    """
    prompt = f"""
    Você é um especialista em acessibilidade e conhece as regras WCAG.
    Baseado na pergunta do usuário:

    '{pergunta}'

    E na seguinte regra relevante:

    {regra_relevante["description"]} - {regra_relevante["details"]}

    Forneça uma resposta clara e detalhada.
    """
    response = generate(model="llama3.1:8b", prompt=prompt)

    # Substitua "response" pela chave correta com base no teste
    return response["response"]

def main():
    # Pergunta ao usuário
    pergunta = input("Digite sua pergunta sobre as regras WCAG: ")

    # Encontra a regra mais relevante
    regra_relevante = encontrar_regra_relevante(pergunta)

    # Consulta o modelo do Ollama
    resposta = consultar_ollama(pergunta, regra_relevante)

    # Mostra a resposta
    print("\n--- Resposta do Especialista ---")
    print(resposta)

if __name__ == "__main__":
    main()