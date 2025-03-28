from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import chainlit as cl
import os
import io

# Chainlit busca variáveis de ambiente do .env automaticamente

""" from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
 """

# Cria um objeto OpenAI
llm = OpenAI()


def create_agent(data: str, llm):
    """Cria um agente DataFrame Pandas."""
    return create_pandas_dataframe_agent(llm, data, verbose=False)


@cl.on_chat_start
async def on_chat_start():

    # Enviando uma imagem com o caminho do arquivo local
    elements = [
    cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(content="Olá, bem-vindo ao AskAnyQuery relacionado a Dados!", elements=elements).send()

    files = None

    # Aguarda o usuário fazer upload dos dados csv
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, faça upload de um arquivo csv para começar!", 
            accept=["text/csv"],
            max_size_mb= 100,
            timeout = 180,
        ).send()

    # carrega os dados csv e armazena na sessão do usuário
    file = files[0]

    msg = cl.Message(content=f"Processando `{file.name}`...")
    await msg.send()

    # Lê arquivo csv com pandas
    csv_file = io.BytesIO(file.content)
    df = pd.read_csv(csv_file, encoding="utf-8")

    # criando sessão do usuário para armazenar dados
    cl.user_session.set('data', df)

    # Envia resposta de volta ao usuário
    # Avisa ao usuário que o sistema está pronto
    msg.content = f"Processamento de `{file.name}` concluído. Você já pode fazer perguntas!"
    await msg.update()


@cl.on_message
async def main(message: str):

    # Obtém dados
    df = cl.user_session.get('data')

    # Criação do agente
    agent = create_agent(df, llm)

    # Executa o modelo
    response = agent.run(message)

    # Envia uma resposta de volta ao usuário
    await cl.Message(
        content=response,
    ).send()