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
from chainlit.element import Text
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import unicodedata
import re
import logging
import json
from datetime import datetime

from pdf2image import convert_from_bytes
import pytesseract
import cv2
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, filename="chat_pericial.log", filemode="a", format="%(asctime)s - %(levelname)s - %(message)s")

CHAT_HISTORY_FILE = "chat_history.json"

# Metadados extraídos manualmente
extracted_metadata = {}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

def save_chat_history(user_input, response):
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "response": response
    }
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append(history_entry)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

nlp = spacy.load("pt_core_news_md")
sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes)
    full_text = ""
    for img in images:
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(image, lang="por")
        full_text += text + "\n"
    return full_text

def extract_named_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def rerank_semantically(question: str, documents: list[Document]) -> list[Document]:
    doc_texts = [doc.page_content for doc in documents]
    doc_embeddings = sbert_model.encode(doc_texts)
    question_embedding = sbert_model.encode([question])[0]
    scores = cosine_similarity([question_embedding], doc_embeddings)[0]
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

def extract_explicit_metadata(text: str) -> dict:
    metadata = {}

    # Informações básicas
    if match := re.search(r"Data da Autua[çc][aã]o[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_autuacao"] = match.group(1)
    if match := re.search(r"Valor da causa[:\s]+R\$\s*([\d.,]+)", text, re.IGNORECASE):
        metadata["valor_causa"] = match.group(1)
    if match := re.search(r"Processo\s*n[oº]?[:\s]*(\d{7}-\d{2}\.\d{4}\.\d{1,2}\.\d{4})", text):
        metadata["numero_processo"] = match.group(1)
    if match := re.search(r"Vara do Trabalho de ([\w\s]+)", text):
        metadata["vara"] = match.group(1).strip()

    # Partes
    if match := re.search(r"Reclamante[:\s]+([\w\s\.]+)", text):
        metadata["reclamante"] = match.group(1).strip()
    if match := re.search(r"Reclamada[:\s]+([\w\s\.]+)", text):
        metadata["reclamada"] = match.group(1).strip()

    # Perícia médica
    if match := re.search(r"M[eé]dico Perito[:\s\n]+([\w\s\.]+)", text):
        metadata["medico_perito"] = match.group(1).strip()
    if match := re.search(r"Per[íi]cia (ser[aá]|foi) realizada em[:\s\n]*([\d/]+)[\s\n]*[\s\n]*([\d:]+)?", text):
        metadata["data_pericia"] = match.group(2)
        if match.group(3):
            metadata["hora_pericia"] = match.group(3)

    # Local e cidade
    if match := re.search(r"local da per[íi]cia[:\s]*([\w\s,º\.\-]+)", text, re.IGNORECASE):
        metadata["local_pericia"] = match.group(1).strip()
    if match := re.search(r"Vara do Trabalho de ([\w\s]+)", text):
        metadata["cidade_vara"] = match.group(1).strip()

    # Função, setor, horários
    if match := re.search(r"fun[çc][aã]o.*?reclamante[:\s\-]*([\w\s]+)", text, re.IGNORECASE):
        metadata["funcao_reclamante"] = match.group(1).strip()
    if match := re.search(r"setor.*?trabalhava[:\s\-]*([\w\s]+)", text, re.IGNORECASE):
        metadata["setor"] = match.group(1).strip()
    if match := re.search(r"escala.*?hor[áa]rio.*?trabalho[:\s\-]*([\w\s]+)", text, re.IGNORECASE):
        metadata["horario_trabalho"] = match.group(1).strip()

    # Admissão e demissão
    if match := re.search(r"admitido em[:\s]*([\d/]+)", text, re.IGNORECASE):
        metadata["data_admissao"] = match.group(1)
    if match := re.search(r"demitido em[:\s]*([\d/]+)", text, re.IGNORECASE):
        metadata["data_demissao"] = match.group(1)

    return metadata

def format_metadata_for_prompt(metadata: dict) -> str:
    if not metadata:
        return "Nenhum metadado detectado."
    return "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in metadata.items()])

def build_prompt_with_metadata(metadata: dict):
    metadata_str = format_metadata_for_prompt(metadata)
    system_template = f"""
Você é um assistente jurídico inteligente. Use exclusivamente as informações fornecidas no contexto para responder às perguntas do usuário.

Se a resposta não estiver presente no conteúdo, diga:
\"Não encontrei essa informação no documento.\"

Responda de forma objetiva, clara e precisa, considerando o ponto de vista técnico de um agente pericial.

Metadados extraídos automaticamente do documento:
{metadata_str}
----------------
{{summaries}}
"""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

# Visualização interativa dos metadados no Chainlit
async def show_extracted_metadata(metadata: dict):
    if metadata:
        lines = ["| Campo | Valor |", "|-------|-------|"]
        lines += [f"| {k.replace('_', ' ').capitalize()} | {v} |" for k, v in metadata.items()]
        await cl.Message(content="🗂️ Metadados extraídos automaticamente do documento:\n" + "\n".join(lines)).send()
    else:
        await cl.Message(content="⚠️ Nenhum metadado foi detectado automaticamente.").send()



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
        source_method = "PyPDF2"
        if not pdf_text.strip():
            raise ValueError("Texto extraído vazio. Usando OCR.")
    except Exception as e:
        logging.warning(f"[PDF EXTRACTION] PyPDF2 falhou: {e}")
        try:
            pdf_text = extract_text_with_ocr(pdf_bytes)
            source_method = "OCR"
        except Exception as ocr_e:
            logging.error(f"[OCR ERROR] {ocr_e}")
            await cl.Message(content=f"Erro ao processar o PDF: {str(ocr_e)}").send()
            return

    global extracted_metadata
    extracted_metadata = extract_explicit_metadata(pdf_text)
    logging.info(f"[METADATA EXTRAIDA] {extracted_metadata}")

    pdf_text = pdf_text.replace("-\n", "").replace("\n", " ")
    original_chunks = text_splitter.split_text(pdf_text)
    normalized_texts = [normalize_text(t) for t in original_chunks]
    metadatas = [{"source": f"Trecho {i+1}"} for i in range(len(normalized_texts))]

    logging.info(f"[EXTRACTION] Método: {source_method}, Chunks gerados: {len(original_chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = await cl.make_async(Chroma.from_texts)(normalized_texts, embeddings, metadatas=metadatas)
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": build_prompt_with_metadata(extracted_metadata)
        }
    )

    cl.user_session.set("chain", chain)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("original_texts", original_chunks)
    cl.user_session.set("normalized_texts", normalized_texts)

    msg.content = f"Processamento de `{file.name}` concluído com sucesso via `{source_method}`! Pode perguntar algo. 📄"
    await msg.update()

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="⚠️ Cadeia não inicializada. Envie um PDF para começar.").send()
        return

    query = normalize_text(message.content)
    docs = await chain.retriever.ainvoke(query)

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

        logging.info("[TRECHOS USADOS] " + "; ".join([doc.metadata.get("source", "?") for doc in original_docs]))

        res = await chain.ainvoke({"question": message.content})

        if isinstance(res, dict):
            answer = res.get("answer") or "Resposta não encontrada."
        else:
            answer = str(res)

        save_chat_history(message.content, answer)

    except Exception as e:
        logging.error(f"[LLM ERROR] {e}")
        answer = f"Erro ao gerar resposta: {str(e)}"

    await cl.Message(content=answer.strip()).send()