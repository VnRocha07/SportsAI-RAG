import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def perguntar_deepseek(prompt):
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system",
             "content": "Responda de forma objetiva usando apenas o contexto fornecido."
             "Não faça inferências sobre datas ou fatos que não estejam explicitamente no contexto."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "disabled"}},
    )

    return response.choices[0].message.content
