# Documentação do SportsAI RAG

> **Status do projeto:** protótipo de estudos em desenvolvimento.  
> **Objetivo:** estudar e demonstrar, de forma prática, um fluxo de RAG aplicado a informações sobre futebol brasileiro.  
> **Escopo atual da base:** conteúdo histórico e contextual entre 2020 e 04 de setembro de 2026.

## Visão geral

O SportsAI é um projeto de estudo que utiliza uma base documental em PDF para responder perguntas sobre futebol. O sistema não foi desenvolvido como produto final, mas sim com o objetivo de servir como estudo na construção de um pipeline de recuperação de contexto, busca semântica e geração de respostas com um modelo de linguagem.

A implementação atual é executada pelo terminal. Os documentos são processados localmente, divididos em trechos menores, transformados em embeddings e armazenados no ChromaDB. Quando o usuário realiza uma pergunta, o sistema procura os trechos semanticamente mais próximos e envia somente o contexto recuperado para a API do DeepSeek.

O fluxo principal é:

```text
PDF > Extração do texto > Documento Langchain > Divisão em Chunks >  Embeddings Locais > ChromaDB > Busca Semântica > Selecionar 5 chunks mais relevantes > Montar Prompt > API Deepseek > Resposta em streaming.
```

## Base de conhecimento

A base atual está localizada em src/base/. O carregamento é feito pelo arquivo src/criar_db.py, que procura todos os arquivos com extensão .pdf dentro dessa pasta.

A leitura dos PDFs utiliza o pypdf. Cada página é extraída separadamente e convertida em um documento do Langchain, para salvar textos e metadados úteis para identificar a origem do conteúdo como podemos ver abaixo:

```text
source > caminho do arquivo
file_name > nome do PDF
page > número da página
start_index > osição inicial do chunk no texto original
```

A base utilizada atualmente contém informações relacionadas ao futebol brasileiro entre 2020 e 04 de Setembro de 2026. Como se trata de uma fonte estática, informações que mudam com frequência podem ficar desatualizadas. O projeto não possui acesso à internet ou atualização em tempo real.

## Divisão dos documentos em chunks

Depois da extração, os documentos são divididos com RecursiveCharacterTextSplitter, sendo configurado da seguinte maneira abaixo:

```python
chunk_size=1000
chunk_overlap=200
length_function=len
add_start_index=True
```

Como o length_function está configurado para receber a função len, os parâmetros chunk_size e chunk_overlap devem trabalhar com caracteres, ou seja, cada chunk deverá ter 1000 caracteres e o próximo chunk deverá ter 200 caracteres do chunk anterior. É possível atribuir uma função de contabilizar tokens no lugar da função len, dessa forma, chunk_size e chunk_overlap poderiam trabalhar com token, não com caracteres.

Esses valores foram definidos e ajustados de forma experimental durante o desenvolvimento, não sendo obrigatório o uso dos mesmos valores. Sempre que os parâmetros de chunking forem alterados, o banco vetorial deve ser recriado.

## Modelo de embeddings

O projeto utiliza do modelo sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, carregado por meio da HuggingFaceEmbeddings, que faz parte da integração langchain-huggingface. Sendo escolhido por ser multilíngue, com suporte a português, transformando sentenças e parágrafos em vetores.

Os embeddings são executados localmente, a base de dados e as perguntas não precisam ser enviados ao Hugging Face para que os vetores sejam calculados. Todavia, na primeira vez que for rodar, o modelo precisa ser baixado para ser armazenado em cache na máquina.

## Banco de dados vetorial

Os embeddings dos chunks são armazenados via ChromaDB, utilizando a integração langchain-chroma, ela ocorre com persistência local:

```python
Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db"
)
```

Após isso, é gerada uma nova pasta que vai conter os metadados guardados no sqlite3 e os vetores guardados no ChromaDB, como o DB é criado a partir da base de dados, não deve ser versionado no git. Portanto, ao clonar o projeto em uma nova máquina, o banco precisa ser criado novamente.

```text
db/
```

Também é recomendado recriar o banco sempre que houver alteração em PDFs da base, chunk_size, chunk_overlap, modelo de embeddings ou configuração de normalização.

Para uma reconstrução limpa, deve-se remover a pasta db/ antiga antes de executar novamente src/criar_db.py.

## Recuperação do contexto

A busca é realizada em src/main.py com:

```python
db.similarity_search_with_relevance_scores(pergunta, k=5)
```

O valor de k representa os chunks recuperados da base de dados para compor o contexto enviado ao modelo de IA, esse valor foi escolhido como um equilíbrio entre recuperar contexto suficiente e evitar o envio de muitos trechos desnecessários para a API. Valores maiores podem aumentar a chance de encontrar uma informação relevante para o contexto, já que são enviados mais chunks. Todavia, valores maiores podem aumentar o gasto de tokens e o prompt para o modelo.

Também um valor de similaridade entre a pergunta do usuário e os chunks recuperados, a similaridade é normalizada para ser entre 0 até 1, quanto mais próximo de 1, maior a similaridade da pergunta do usuário com o chunk recuperado.

Com base nas informações acima, podemos assumir que o valor de similaridade está diretamente conectado com a relevância daquele chunk de responder uma pergunta. Ou seja, se o valor do primeiro chunk (que possui a maior nota) for muito baixo, o sistema pode alucinar ou não conseguir responder a pergunta. Portanto, levando isso em consideração, foi incluido uma checagem, se o primeiro chunk ter uma nota menor que 0.2, o sistema não envia a consulta ao DeepSeek, dessa maneira, podemos economizar tokens.

Vale lembrar que é possível obter valores abaixo de 0, mas estes geralmente se devem a perguntas fora do tópico de futebol.

## Construção do prompt

Os chunks recuperados são unidos utilizando "---" como separador, o conteúdo dos 5 chunks recuperados é guardado na base_conhecimento, uma variável que será utilizada para montar o prompt principal do sistema.

O prompt atual coloca primeiro instruções básicas, seguido pelo contexto recuperado e depois a pergunta do usuário. A instrução principal exige que a resposta seja produzida somente a partir das informações fornecidas pela recuperação.

Essa abordagem busca reduzir respostas baseadas exclusivamente no conhecimento prévio do modelo e tornar mais fácil identificar quando a base documental não contém a informação necessária.

## Modelo generativo e API

A geração da resposta utiliza a API do DeepSeek por meio do SDK da OpenAI. Por motivos de segurança, a chave da API do DeepSeek não é versionada, é necessário cada usuário ter sua própria chave de API.

O programa permite selecionar pelo terminal, entre dois modelos da DeepSeek, sendo eles o DeepSeek V4 Pro e DeepSeek V4 Flash.

O projeto usa stream=True, portanto a resposta é impressa gradualmente no terminal conforme a saída chegam da API.

No estado atual, o modo thinking está desabilitado.

A chamada também mantém reasoning_effort="high". Essa configuração faz parte do estado atual do protótipo e pode ser revisada futuramente durante novos testes de custo, latência e qualidade.

O prompt do sistema pede respostas objetivas e contextualizadas, proibindo interferências sobre fatos ou datas que não estejam explicitamente presentes no contexto e inclui uma instrução básica para não revelar informações privadas ou credenciais para casos futuros.

## Estrutura principal do projeto

A estrutura relevante para o RAG é aproximadamente:

```text
SportsAI-RAG/
├── src/
│   ├── base/
│   │   └── arquivos PDF
│   ├── api.py
│   ├── criar_db.py
│   └── main.py
├── .env
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

Depois da indexação, também é criada localmente uma pasta db/ para guardar os metadados e os embeddings.

O script src/criar_db.py é responsável por carregar os PDFs, criar os documentos LangChain, dividir o conteúdo em chunks, gerar embeddings e guardar os vetores no ChromaDB.

Já o src/main.py recebe a pergunta, carrega o mesmo modelo de embeddings, abre o ChromaDB, executa a busca semântica, verifica a similaridade, monta o contexto e encaminha o prompt para a API.

A src/api.py centraliza a comunicação com o DeepSeek e imprime a resposta em streaming.

## Dependências principais

O projeto utiliza Python 3.14 e gerencia as dependências com uv. As dependências principais registradas no pyproject.toml são:

```text
chromadb
langchain
langchain-chroma
langchain-huggingface
langchain-text-splitters
openai
pypdf
python-dotenv
sentence-transformers
```

## Executando o projeto

### Etapa 1

É necessário ter Git, Python 3.14, uv, uma chave válida da API do DeepSeek e conexão com a internet para a API e para o download inicial do modelo de embeddings.

Primeiramente, precisamos clonar o repositório.

```bash
git clone -b Hotfix https://github.com/VnRocha07/SportsAI-RAG.git
cd SportsAI-RAG
```

### Etapa 2

Depois, é necessário sincronizar as dependências com o ambiente virtual utilizando o comando abaixo:

```bash
uv sync
```

O uv utilizará pyproject.toml e uv.lock para preparar o ambiente.

### Etapa 3

É preciso configurar a chave de API do DeepSeek para utilizar o modelo, crie um arquivo chamado .env na raiz do projeto:

```env
DEEPSEEK_API_KEY=sua_chave_aqui
```
O .env está listado no .gitignore e não deve ser enviado ao repositório.

### Etapa 4

Antes de criar o banco de dados vetorial, deve ser confirmar se a base de dados está em src/base/ pois o carregador processa todos os arquivos PDF encontrados nessa pasta.

Na primeira execução, ou sempre que a base, o chunking ou os embeddings forem alterados, execute:

```bash
uv run src/criar_db.py
```

Essa etapa lê os PDFs, divide os documentos, carrega o modelo de embeddings, cria os vetores e grava o ChromaDB em db/.

Na primeira vez, o modelo do Hugging Face pode levar algum tempo para ser baixado. Nas execuções seguintes ele normalmente será carregado do cache local.

### Etapa 5

Depois que o banco tiver sido criado, será possível rodar o programa:

```bash
uv run src/main.py
```

## OBSERVAÇÃO

Não é necessário executar criar_db.py antes de toda pergunta. O banco persiste em db/ e pode ser reutilizado. Ele deve ser recriado quando houver mudanças que alterem os vetores ou os documentos indexados. O procedimento recomendado é:

```text
1. remover a pasta db/
2. aplicar as alterações desejadas
3. executar uv run src/criar_db.py
4. executar uv run src/main.py
```

## Limitações atuais

O SportsAI é apenas um sistema básico feito para estudar e práticar sobre RAG, não deve ser tratado como uma fonte oficial.

A base de dados é estática e não contém informações posteriores ao período de atualização (04/09/2026). A recuperação semântica também não garante que o melhor trecho possível esteja entre os primeiros resultados.

O projeto atualmente não possui histórico de conversa ou interface gráfica, ambas as funcionalidades estão sendo trabalhadas.
