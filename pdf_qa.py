# Import necessary modules and define env variables
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document

import os
import chainlit as cl
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import unicodedata
import re

load_dotenv()

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

system_template = """
Você é um assistente jurídico inteligente. Use exclusivamente as informações fornecidas no contexto para responder às perguntas do usuário.

Se a resposta não estiver presente no conteúdo, diga:
"Não encontrei essa informação no documento."

Responda de forma objetiva e concisa, sem mencionar fontes ou trechos.
----------------
{summaries}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

async def rerank_contexts(question: str, documents: list[Document], model_name: str = "gpt-4") -> list[Document]:
    llm = ChatOpenAI(model=model_name, temperature=0)
    context_list = "\n\n".join(
        [f"Trecho {i+1}:\n\"\"\"\n{doc.page_content}\n\"\"\"" for i, doc in enumerate(documents)]
    )
    rerank_prompt = f"""
Abaixo estão trechos de um documento e uma pergunta feita por um usuário.

Pergunta: "{question}"

Classifique os trechos abaixo por relevância para responder à pergunta. Liste os trechos do mais relevante ao menos relevante, usando os números.

Trechos:
{context_list}

Responda com uma lista separada por vírgulas, como: 2, 1, 3, 4
"""
    try:
        response = await llm.ainvoke(rerank_prompt)
        content = response.content if hasattr(response, 'content') else response
        order = [int(i.strip()) - 1 for i in content.split(",") if i.strip().isdigit()]
        ranked_docs = [documents[i] for i in order if 0 <= i < len(documents)]
        return ranked_docs
    except Exception as e:
        print(f"[RERANKING ERROR] {e}")
        return documents

@cl.on_chat_start
async def on_chat_start():
    elements = [cl.Image(name="image1", display="inline", path="./robot.jpeg")]
    await cl.Message(content="Olá! Bem-vindo ao Chat Pericial! Envie um PDF para começar. 🤖", elements=elements).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, envie um arquivo PDF para começarmos.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processando `{file.name}`...")
    await msg.send()

    try:
        with open(file.path, "rb") as f:
            pdf_bytes = f.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        pdf_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    except Exception as e:
        await cl.Message(content=f"Erro ao processar o PDF: {str(e)}").send()
        return

    pdf_text = pdf_text.replace("-\n", "").replace("\n", " ")
    normalized_texts = [normalize_text(t) for t in text_splitter.split_text(pdf_text)]
    metadatas = [{"source": f"Trecho {i+1}"} for i in range(len(normalized_texts))]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = await cl.make_async(Chroma.from_texts)(normalized_texts, embeddings, metadatas=metadatas)
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs
    )

    cl.user_session.set("chain", chain)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("original_texts", text_splitter.split_text(pdf_text))
    cl.user_session.set("normalized_texts", normalized_texts)

    msg.content = f"Processamento de `{file.name}` concluído! Pode perguntar algo. 📄"
    await msg.update()

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="⚠️ Cadeia não inicializada. Envie um PDF para começar.").send()
        return

    retriever = chain.retriever
    query = normalize_text(message.content)
    docs = await retriever.ainvoke(query)

    if not docs:
        await cl.Message(content="Nenhum trecho relevante encontrado no documento para essa pergunta.").send()
        return

    try:
        original_texts = cl.user_session.get("original_texts", [])
        metadatas = cl.user_session.get("metadatas", [])

        original_docs = []
        for doc in docs:
            try:
                index = metadatas.index(doc.metadata)
                text = original_texts[index]
                original_docs.append(Document(page_content=text, metadata=doc.metadata))
            except:
                original_docs.append(doc)

        res = await chain.ainvoke({"question": message.content})

        if isinstance(res, dict):
            answer = res.get("answer") or "Resposta não encontrada."
        else:
            answer = str(res)
    except Exception as e:
        answer = f"Erro ao gerar resposta: {str(e)}"

    await cl.Message(content=answer.strip()).send()
