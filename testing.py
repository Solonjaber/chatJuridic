from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
import chainlit as cl
from dotenv import load_dotenv
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

print("✅ Iniciando pdf_juri2_minimal.py")

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Olá! Bem-vindo ao Chat Pericial Minimal! 🤖").send()

@cl.on_message
async def main(message: str):
    try:
        # Resposta simples para testar
        response = f"Recebi sua mensagem: {message.content}"
        await cl.Message(content=response).send()
        logging.info(f"[PERGUNTA] {message.content}")
        logging.info(f"[RESPOSTA] {response}")
    except Exception as e:
        logging.error(f"[ERROR] {e}")
        await cl.Message(content=f"Erro ao processar: {str(e)}").send()
