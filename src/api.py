import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def perguntar_deepseek(prompt, modelo):
    response = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system",
             "content": "Responda de forma objetiva e detalhada usando apenas o contexto fornecido."
             "Não faça inferências sobre datas ou fatos que não estejam explicitamente no contexto."},
             {"Nunca repasse informações da base de conhecimento ou informações privadas, mesmo que solicitadas."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "disabled"}},
    )

    resposta_completa = ""

    for chunk in response:
        conteudo = chunk.choices[0].delta.content

        if conteudo:
            print(conteudo, end="", flush=True)
            resposta_completa += conteudo

    print()

    return resposta_completa
