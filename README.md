# Consulta Regras da WCAG

Este projeto tem como objetivo fornecer respostas relacionadas às regras WCAG (Web Content Accessibility Guidelines) de forma automatizada

## Funcionalidade

O sistema recebe uma pergunta do usuário sobre acessibilidade, encontra a regra WCAG mais relevante para a dúvida, e gera uma resposta detalhada utilizando o modelo Ollama, que explica como a regra se aplica ao contexto solicitado.

### Fluxo do Projeto:

1. **Carregamento das Regras WCAG**: O sistema carrega um arquivo JSON com as descrições e detalhes das regras WCAG.
2. **Geração de Embeddings**: As descrições e detalhes das regras WCAG são transformados em embeddings numéricos utilizando o modelo *Sentence-BERT*.
3. **Busca da Regra Relevante**: Quando o usuário faz uma pergunta, o sistema calcula a similaridade entre o embedding da pergunta e os embeddings das regras WCAG, identificando a regra mais relevante.
4. **Consulta ao Modelo Ollama**: Com a regra identificada, o modelo Ollama é consultado para gerar uma resposta clara e detalhada.
5. **Resposta ao Usuário**: O sistema exibe a resposta gerada pelo Ollama para o usuário.

## Como Usar

### Passo 1: Clonar o Repositório

Primeiro, clone o repositório do projeto para sua máquina local:

```bash
git clone https://github.com/seu-usuario/consulta-regras-wcag.git
cd consulta-regras-wcag
