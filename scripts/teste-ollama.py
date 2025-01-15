from ollama import generate

response = generate(model="llama3.1:8b", prompt="Qual é a capital da França?")
print(response)