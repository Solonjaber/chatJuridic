from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl


from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use as seguintes partes do contexto para responder à pergunta do usuário.
Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.
SEMPRE retorne uma parte "FONTES" em sua resposta.
A parte "FONTES" deve ser uma referência à fonte do documento de onde você obteve sua resposta.

Exemplo de sua resposta deve ser:


A resposta é foo
FONTES: solon


Comece!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():
    # Enviando uma imagem com o caminho do arquivo local
    elements = [
    cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(content="Olá, Bem-vindo ao AskAnyQuery relacionado a Dados!", elements=elements).send()
    files = None

    # Aguarde o usuário fazer upload de um arquivo
    while files == None:
        files = await cl.AskFileMessage(
            content="Por favor, faça upload de um arquivo de texto para começar!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processando `{file.name}`...")
    await msg.send()

    # Decodifica o arquivo
    text = file.content.decode("utf-8")

    # Divide o texto em partes
    texts = text_splitter.split_text(text)

    # Cria metadados para cada parte
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Cria um armazenamento de vetores Chroma
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    # Cria uma cadeia que usa o armazenamento de vetores Chroma
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Salva os metadados e textos na sessão do usuário
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Avisa ao usuário que o sistema está pronto
    msg.content = f"Processamento de `{file.name}` concluído. Você já pode fazer perguntas!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["RESPOSTA", "FINAL"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Obtém os metadados e textos da sessão do usuário
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Adiciona as fontes à mensagem
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Obtém o índice da fonte
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Cria o elemento de texto referenciado na mensagem
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nFontes: {', '.join(found_sources)}"
        else:
            answer += "\nNenhuma fonte encontrada"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()