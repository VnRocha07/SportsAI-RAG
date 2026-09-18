from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from api import perguntar_deepseek

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """
Você é um assistente, analise a base de conhecimento abaixo para responder à próxima pergunta.

{base_conhecimento}

Responda APENAS com base nas informações acima:

{pergunta}"""


def perguntar():
    escolher_modelo = input("Selecione o modelo: \n 1-Pro \n 2-Flash \n")

    modelos = {"1": "deepseek-v4-pro", "2": "deepseek-v4-flash"}

    modelo = modelos.get(escolher_modelo, lambda: print(
        "Operação inválida, selecione um modelo."))

    while True: #Fiz só esse while true pra conversa ficar continua, depois do refactor vou add essa feature
        pergunta = input("Pergunta: ")
        if pergunta.lower().strip() == "sair":
            print("Encerrando o programa.")
            break

        funcao_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        db = Chroma(
            persist_directory=CAMINHO_DB,
            embedding_function=funcao_embeddings
        )

        resultados = db.similarity_search_with_relevance_scores(pergunta, k=7)
        if len(resultados) == 0 or resultados[0][1] < 0.2:
            print("Não consegui encontrar nenhuma informação relevante na base")
            return

        textos_resultado = []
        for resultado in resultados:
            texto = resultado[0].page_content
            textos_resultado.append(texto)

        base_conhecimento = "\n\n----\n\n".join(textos_resultado)
        prompt = PromptTemplate.from_template(prompt_template)
        prompt_formatado = prompt.invoke({
            "pergunta": pergunta,
            "base_conhecimento": base_conhecimento
        })

        perguntar_deepseek(prompt_formatado.text, modelo)


perguntar()
