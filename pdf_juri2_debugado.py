from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
import chainlit as cl
from chainlit.element import Element
from chainlit.action import Action
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
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
#     handlers=[
#         logging.FileHandler("chat_pericial.log", mode='a', encoding="utf-8"),
#         logging.StreamHandler()  # <-- Exibe no console do Railway
#     ]
# )


def log_similarity_scores(query: str, docs: list, scores: list):
    
    log_data = {
        "query": query,
        "top_results": [
            {
                "source": doc.metadata.get("source", "?"),
                "score": float(score),
                "preview": doc.page_content[:50] + "..."
            } for doc, score in zip(docs, scores)
        ]
    }
    logging.info(f"[SIMILARITY SCORES] {json.dumps(log_data, ensure_ascii=False)}")

CHAT_HISTORY_FILE = "chat_history.json"


extracted_metadata = {}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])

# def save_chat_history(user_input, response):
#     history_entry = {
#         "timestamp": datetime.now().isoformat(),
#         "user_input": user_input,
#         "response": response
#     }
#     if os.path.exists(CHAT_HISTORY_FILE):
#         with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
#             history = json.load(f)
#     else:
#         history = []
#     history.append(history_entry)
#     with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2, ensure_ascii=False)

# nlp = spacy.load("pt_core_news_md")

# try:
#     logging.info("⚠️ SBERT desativado para teste no Railway")
#     sbert_model = None
# except Exception as e:
#     logging.error(f"❌ Erro ao carregar SentenceTransformer: {e}")
#     import sys
#     sys.stderr.write(f"[SBERT ERROR] {str(e)}\n")

@cl.action_callback("load_chat_history")
async def load_chat_history(action):
    
    chat_id = action.payload
    
    history = load_chat_history_from_storage(chat_id)
    
    if history:
        
        await reset_user_session()
        
        
        for msg in history["messages"]:
            if msg["role"] == "user":
                await cl.Message(content=msg["content"], author="Usuário").send()
            else:
                await cl.Message(content=msg["content"]).send()
        
        await cl.Message(content="Histórico de conversa restaurado.").send()
    else:
        await cl.Message(content="Não foi possível carregar o histórico.").send()

def load_chat_history_from_storage(chat_id):
    
    try:
        with open(f"chat_history{chat_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def process_pdf_with_hybrid_extraction(pdf_bytes: bytes) -> str:
    
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    full_text = ""
    extraction_methods = []
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        
        
        if not page_text.strip() or len(page_text.strip()) < 100:
            logging.info(f"[OCR] Aplicando OCR na página {i+1} devido a texto insuficiente")
            try:
                
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                if images:
                    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
                    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
                    ocr_text = pytesseract.image_to_string(image, lang="por")
                    page_text = ocr_text
                    extraction_methods.append(f"Página {i+1}: OCR")
                else:
                    extraction_methods.append(f"Página {i+1}: Falha na conversão para imagem")
            except Exception as e:
                logging.error(f"[OCR ERROR] Página {i+1}: {e}")
                extraction_methods.append(f"Página {i+1}: Erro OCR")
        else:
            extraction_methods.append(f"Página {i+1}: PyPDF2")
        
        full_text += page_text + "\n\n"
    
    logging.info(f"[EXTRACTION METHODS] {'; '.join(extraction_methods)}")
    return full_text

# def detect_document_type(text: str) -> str:

    
    
    keywords = {
        "petição_inicial": ["petição inicial", "autor requer", "dos pedidos", "dos fatos", "do direito", 
                           "deferimento", "termos em que", "pede deferimento"],
        "contestação": ["contestação", "preliminarmente", "mérito", "improcedente", "improcedência", 
                       "contesta", "contestar"],
        "laudo_pericial": ["laudo pericial", "perícia", "perito", "quesitos", "vistoria", "exame", 
                          "conclusão técnica", "metodologia"],
        "sentença": ["sentença", "julgo", "dispositivo", "condeno", "improcedente", "procedente", 
                    "fundamentação", "relatório", "isto posto"],
        "despacho": ["despacho", "intime-se", "cite-se", "certifique-se", "cumpra-se", "manifeste-se"],
        "acórdão": ["acórdão", "votação", "turma", "câmara", "relator", "revisor", "ementa"]
    }
    
    
    counts = {doc_type: 0 for doc_type in keywords}
    text_lower = text.lower()
    
    for doc_type, terms in keywords.items():
        for term in terms:
            counts[doc_type] += text_lower.count(term)
    
    
    doc_types = list(keywords.keys())
    doc_descriptions = [
        "Petição inicial com pedidos e fatos",
        "Contestação com argumentos de defesa",
        "Laudo pericial com análise técnica",
        "Sentença judicial com decisão",
        "Despacho com determinações processuais",
        "Acórdão com decisão colegiada"
    ]
    
    
    # text_embedding = # sbert_model.encode([text_lower[:1000]])[0]
    # desc_embeddings = # sbert_model.encode(doc_descriptions)
    
    
    similarities = cosine_similarity([text_embedding], desc_embeddings)[0]
    
    
    combined_scores = {
        doc_type: (counts[doc_type] * 0.7) + (similarities[i] * 0.3)
        for i, doc_type in enumerate(doc_types)
    }
    
    
    most_likely_type = max(combined_scores, key=combined_scores.get)
    confidence = combined_scores[most_likely_type]
    
    logging.info(f"[DOCUMENT TYPE] Detectado: {most_likely_type} (confiança: {confidence:.2f})")
    
    return most_likely_type

def expand_question_for_legal_context(question: str) -> str:
    synonyms = {
        "autor": ["reclamante", "parte autora", "requerente", "demandante", "nome do autor", "quem é o autor"],
        "réu": ["reclamada", "empresa", "demandado", "parte ré", "nome do réu", "quem é o réu"],
        "advogado": ["procurador", "representante legal", "oab", "defensor", "advogado da parte"],
        "perito": ["especialista", "médico perito", "engenheiro", "assistente técnico"]
    }

    generic_terms = ["nome", "quem é", "qual o nome", "identificação"]

    expanded = question.lower()

    if any(term in expanded for term in generic_terms):
        
        expanded += " autor reclamante réu reclamada parte"


    for key, terms in synonyms.items():
        if key in expanded:
            expanded += " " + " ".join(terms)
    return expanded

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

def process_pdf_with_hybrid_extraction(pdf_bytes: bytes) -> str:
    
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    full_text = ""
    extraction_methods = []
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        
        
        if not page_text.strip() or len(page_text.strip()) < 100:
            logging.info(f"[OCR] Aplicando OCR na página {i+1} devido a texto insuficiente")
            try:
                
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                if images:
                    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
                    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
                    ocr_text = pytesseract.image_to_string(image, lang="por")
                    page_text = ocr_text
                    extraction_methods.append(f"Página {i+1}: OCR")
                else:
                    extraction_methods.append(f"Página {i+1}: Falha na conversão para imagem")
            except Exception as e:
                logging.error(f"[OCR ERROR] Página {i+1}: {e}")
                extraction_methods.append(f"Página {i+1}: Erro OCR")
        else:
            extraction_methods.append(f"Página {i+1}: PyPDF2")
        
        full_text += page_text + "\n\n"
    
    logging.info(f"[EXTRACTION METHODS] {'; '.join(extraction_methods)}")
    return full_text

# def extract_named_entities(text: str):
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]

def rerank_semantically(question: str, documents: list[Document]) -> list[Document]:
    
    
    expanded_question = expand_question_for_legal_context(question)
    
    
    if "réu" in question.lower() or "reu" in question.lower():
        expanded_question += " reclamado demandado parte contrária"
    if "autor" in question.lower():
        expanded_question += " reclamante requerente parte autora"
    if "advogado" in question.lower():
        expanded_question += " procurador representante legal oab"
    
    
    # doc_texts = [doc.page_content for doc in documents]
    # doc_embeddings = # sbert_model.encode(doc_texts)
    # question_embedding = # sbert_model.encode([expanded_question])[0]
    
    
    # scores = cosine_similarity([question_embedding], doc_embeddings)[0]
    
    
    # ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    # if max(scores) < 0.4:
    #     return []
    
    
    # logging.info(f"[RERANKING] Pergunta expandida: {expanded_question}")
    # for i, (doc, score) in enumerate(ranked[:3]):
    #     logging.info(f"[RERANKING] Doc {i+1}, Score: {score:.4f}, Preview: {doc.page_content[:80]}...")
        
    # return [doc for doc, _ in ranked]


def extract_explicit_metadata(text: str) -> dict:
    metadata = {}

    
    if match := re.search(r"Data da Autua[çc][aã]o[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_autuacao"] = match.group(1)
    if match := re.search(r"Valor da causa[:\s]+R\$\s*([\d.,]+)", text, re.IGNORECASE):
        metadata["valor_causa"] = match.group(1)
    if match := re.search(r"Processo\s*n[oº]?[:\s]*(\d{7}-\d{2}\.\d{4}\.\d{1,2}\.\d{4})", text):
        metadata["numero_processo"] = match.group(1)
    if match := re.search(r"Vara do Trabalho de ([\w\s]+)", text):
        metadata["vara"] = match.group(1).strip()

    
    match_partes = re.search(
        r"AUTOR[:\s]+([^\n\r]+?)\s+ADVOGADO[:\s]+([^\n\r]+?)\s+R[ÉE]U[:\s]+([^\n\r]+?)\s+ADVOGADO[:\s]+([^\n\r]+)",
        text, re.IGNORECASE
    )
    if match_partes:
        metadata["autor"] = match_partes.group(1).strip()
        metadata["advogado_autor"] = match_partes.group(2).strip()
        metadata["reu"] = match_partes.group(3).strip()
        metadata["advogado_reu"] = match_partes.group(4).strip()
    else:
        linhas = text.splitlines()
        partes = {}
        for i, linha in enumerate(linhas):
            linha_norm = linha.strip().lower()
            if any(p in linha_norm for p in ["autor:", "reclamante:"]):
                partes["autor"] = linha.split(":", 1)[-1].strip()
            elif "advogado" in linha_norm and "autor" in partes and "advogado_autor" not in partes:
                partes["advogado_autor"] = linha.split(":", 1)[-1].strip()
            elif any(p in linha_norm for p in ["réu:", "reclamada:"]):
                partes["reu"] = linha.split(":", 1)[-1].strip()
            elif "advogado" in linha_norm and "reu" in partes and "advogado_reu" not in partes:
                partes["advogado_reu"] = linha.split(":", 1)[-1].strip()
        metadata.update(partes)

    
    if match := re.search(r"OAB[:/\s]*([A-Z]{2}\s*\d+)", text):
        metadata["oab_advogado"] = match.group(1)
    if match := re.search(r"CRM[:/\s]*([A-Z]{2}\s*\d+)", text):
        metadata["crm_perito"] = match.group(1)
    if match := re.search(r"tipo de a[çc][ãa]o[:\s]*([\w\s]+)", text, re.IGNORECASE):
        metadata["tipo_acao"] = match.group(1).strip()
    if match := re.search(r"CPF[:\s]*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})", text, re.IGNORECASE):
        metadata["cpf_reclamante"] = match.group(1)
    if match := re.search(r"CNPJ[:\s]*(\d{2}\.?\d{3}\.?\d{3}/?0001-\d{2})", text, re.IGNORECASE):
        metadata["cnpj_reclamada"] = match.group(1)

    
    # doc = nlp(text)
    # for ent in doc.ents:
    #     if ent.label_ == "PER" and any(term in ent.sent.text.lower() for term in ["juiz", "magistrado", "julgador"]):
    #         metadata["juiz"] = ent.text
    #     if ent.label_ == "LOC" and any(term in ent.sent.text.lower() for term in ["endereço", "localizado", "sede"]):
    #         metadata["endereco_relevante"] = ent.text
    #     if ent.label_ == "LAW" or any(term in ent.text.lower() for term in ["lei", "artigo", "decreto", "clt"]):
    #         if "leis_citadas" not in metadata:
    #             metadata["leis_citadas"] = []
    #         if ent.text not in metadata["leis_citadas"]:
    #             metadata["leis_citadas"].append(ent.text)
    # if "leis_citadas" in metadata and isinstance(metadata["leis_citadas"], list):
    #     metadata["leis_citadas"] = ", ".join(metadata["leis_citadas"])

    # return metadata

def format_metadata_for_prompt(metadata: dict) -> str:
    if not metadata:
        return "Nenhum metadado detectado."
    return "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in metadata.items()])


def build_adaptive_prompt(query: str, metadata: dict):
    
    metadata_str = format_metadata_for_prompt(metadata)
    
    
    query_lower = query.lower()
    
    
    specific_instructions = ""
    
    if any(term in query_lower for term in ["autor", "reclamante", "requerente", "parte"]):
        specific_instructions = """
        Ao responder sobre partes do processo:
        - Se a informação exata **não estiver disponível**, mas houver **qualquer menção relacionada**, **nunca responda apenas "informação não encontrada"**. Explique, com base no documento, o que é mencionado sobre o tema da pergunta.
        - Forneça apenas nomes completos, sem explicações adicionais
        - Se houver qualificação como CPF ou RG, inclua apenas se explicitamente solicitado
        - Seja extremamente conciso
        """
    elif any(term in query_lower for term in ["advogado", "procurador", "representante", "oab"]):
        specific_instructions = """
        Ao responder sobre representantes legais:
        - Forneça apenas o nome e número da OAB, sem explicações adicionais
        - Indique apenas a qual parte o advogado está vinculado, se necessário
        - Seja extremamente conciso
        """
    elif any(term in query_lower for term in ["data", "prazo", "audiência", "perícia"]):
        specific_instructions = """
        Ao responder sobre datas, prazos e perícias:
        - Se a informação exata não estiver disponível, explique o que o documento menciona sobre o assunto
        - Informe sobre determinações, procedimentos ou instruções relacionadas no documento
        - Cite trechos relevantes que mencionem como a informação será definida ou comunicada
        - Seja claro e informativo, mesmo quando a resposta direta não estiver presente
        """
    elif any(term in query_lower for term in ["valor", "causa", "condenação", "indenização", "dano"]):
        specific_instructions = """
        Ao responder sobre valores monetários:
        - Forneça apenas o valor e a que se refere
        - Não inclua explicações sobre juros ou correção, a menos que explicitamente solicitado
        - Seja extremamente conciso
        """
    
    system_template = f"""
    Você é um assistente jurídico especializado em análise de documentos periciais. Use as informações fornecidas no contexto para responder às perguntas do usuário sobre o documento PDF enviado.

    Se a informação estiver presente no documento, forneça uma resposta direta e objetiva.
    
⚠️ Se a informação **não estiver explicitamente presente**, siga esta diretriz:
    - Explique se há previsão, instrução ou citação indireta sobre o tema.
    - Especifique qual parte do documento trata do assunto, mesmo que a resposta não seja conclusiva.
    - Use linguagem precisa e técnica, mas sempre com clareza e empatia pericial.

    IMPORTANTE: Você deve entender o contexto jurídico brasileiro e a terminologia legal. Em documentos jurídicos:
    - Diferentes termos podem se referir às mesmas partes processuais
    - Você deve identificar as partes e seus representantes independentemente da terminologia usada
    - Considere o tipo de documento e o contexto para interpretar corretamente os papéis das pessoas mencionadas
    
    Exemplos do formato esperado:
    Pergunta: "Quem é o autor e o advogado?"
    Resposta: "Autor: O autor principal do documento é Eliane Rodrigues da Silva, e o advogado que a representa é Thiago Kusunoki Ferachin, inscrito na OAB/MS sob o nº 11.645."
    
    Pergunta: "Qual o valor da causa?"
    Resposta: "R$ 50.000,00"
    
    Pergunta: "Quando será realizada a perícia?"
    Resposta: "O documento não especifica a data exata da perícia, mas determina que o perito nomeado (Dr. Carlos Alberto) deverá informar ao Juízo sobre o local, data e horário com antecedência mínima de 10 dias, para que as partes possam ser intimadas e acompanhar os trabalhos periciais."

    Responda de forma objetiva, clara e precisa, considerando o ponto de vista técnico de um agente pericial.

    {specific_instructions}

    Metadados extraídos automaticamente do documento:
    {metadata_str}
    ----------------
    {{context}}
    """
    
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])


async def show_extracted_metadata(metadata: dict):
    if metadata:
        lines = ["| Campo | Valor |", "|-------|-------|"]
        lines += [f"| {k.replace('_', ' ').capitalize()} | {v} |" for k, v in metadata.items()]
        await cl.Message(content="🗂️ Metadados extraídos automaticamente do documento:\n" + "\n".join(lines)).send()
    else:
        await cl.Message(content="⚠️ Nenhum metadado foi detectado automaticamente.").send()


chain_type_kwargs = {
    "prompt": build_adaptive_prompt(query="", metadata={})  
}


async def reset_user_session():
    keys = ["chain", "retriever", "original_texts", "normalized_texts", "metadatas", "metadata"]
    for key in keys:
        if cl.user_session.get(key) is not None:
            cl.user_session.set(key, None)

@cl.on_chat_start
async def on_chat_start():

    elements = [cl.Image(name="image1", display="inline", path="robot.jpeg")]
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
        
        
        pdf_text = process_pdf_with_hybrid_extraction(pdf_bytes)
        source_method = "Híbrido"
        
        if not pdf_text.strip():
            raise ValueError("Não foi possível extrair texto do documento.")
    except Exception as e:
        logging.error(f"[PDF EXTRACTION ERROR] {e}")
        await cl.Message(content=f"Erro ao processar o PDF: {str(e)}").send()
        return

    logging.info("[DOCUMENT TYPE] Detecção desativada para teste")

        
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    global extracted_metadata
    extracted_metadata = extract_explicit_metadata(pdf_text)
    logging.info(f"[METADATA EXTRAIDA] {extracted_metadata}")

    pdf_text = pdf_text.replace("-\n", "").replace("\n", " ")
    original_chunks = text_splitter.split_text(pdf_text)

    
    for i, chunk in enumerate(original_chunks):
        if any(kw in chunk.lower() for kw in ["deverá informar", "com antecedência de", "designar perícia", "será designada", "intimar para perícia"]):
            original_chunks[i] = "[INSTRUÇÃO FUTURA] " + chunk

    normalized_texts = [normalize_text(t) for t in original_chunks]

    metadatas = [{"source": f"Trecho {i+1}"} for i in range(len(normalized_texts))]

    logging.info(f"[EXTRACTION] Método: {source_method}, Chunks gerados: {len(original_chunks)}")

    
    await reset_user_session()
    
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = await cl.make_async(FAISS.from_texts)(
        normalized_texts, embeddings, metadatas=metadatas
    )
    
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}  
    )

    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": build_adaptive_prompt(query="", metadata=extracted_metadata),
            "document_variable_name": "context"
        }
    )

    
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("original_texts", original_chunks)
    cl.user_session.set("normalized_texts", normalized_texts)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("memory", memory)

    msg.content = f"Processamento de `{file.name}` concluído com sucesso via `{source_method}`! Pode perguntar algo. 📄"
    await msg.update()

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="⚠️ Cadeia não inicializada. Envie um PDF para começar.").send()
        return

    
    logging.info(f"[PERGUNTA ORIGINAL] {message.content}")

    query = normalize_text(message.content)
    docs = await chain.retriever.ainvoke(query)

        
    logging.info(f"[DOCUMENTOS RECUPERADOS] Total: {len(docs)}")
    for i, doc in enumerate(docs[:3]):
        logging.info(f"[DOC {i+1}] Fonte: {doc.metadata.get('source', '?')}")
        logging.info(f"[DOC {i+1}] Conteúdo: {doc.page_content[:150]}...")

    # docs = rerank_semantically(message.content, docs)

    if not docs:
        query_terms = " ".join([word for word in message.content.lower().split() if len(word) > 3])
        fallback_docs = await chain.retriever.ainvoke(query_terms)
        
        if fallback_docs:
            docs = fallback_docs[:3]  
            logging.info(f"[FALLBACK] Usando busca alternativa com termos: {query_terms}")
        else:
            await cl.Message(content="Não foi possível encontrar informações relevantes no documento para responder sua pergunta.").send()
            return

    try:
        
        adaptive_prompt = build_adaptive_prompt(message.content, extracted_metadata)
        chain.combine_docs_chain.llm_chain.prompt = adaptive_prompt

        
        if len(docs) > 3:
            docs = docs[:3]
            
        logging.info("[TRECHOS USADOS] " + "; ".join([doc.metadata.get("source", "?") for doc in docs]))
        
        start_time = datetime.now()
        logging.info(f"[PERGUNTA RECEBIDA] {message.content}")
        res = await chain.ainvoke({"question": message.content})
        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"[TEMPO DE RESPOSTA] {response_time:.2f} segundos")

        if isinstance(res, dict):
            answer = res.get("answer") or "Resposta não encontrada."
            
            
            if len(answer) > 2000:
                answer = answer[:1997] + "..."
                
            
            # save_chat_history(message.content, answer)
            
            
            await cl.Message(content=answer).send()
        else:
            answer = str(res)
            if len(answer) > 2000:
                answer = answer[:1997] + "..."
            await cl.Message(content=answer.strip()).send()

        # save_chat_history(message.content, answer)
        # logging.info(f"[PERGUNTA] {message.content}")
        # logging.info(f"[RESPOSTA] {answer[:600]}...")

    except Exception as e:
        logging.error(f"[LLM ERROR] {e}")
        answer = f"Erro ao gerar resposta: {str(e)}"
        await cl.Message(content=answer.strip()).send()
