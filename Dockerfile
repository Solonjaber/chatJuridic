FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm

# Instala bibliotecas de sistema para OCR, PDF e imagem
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala as dependências Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Instala spacy explicitamente
RUN pip install spacy
RUN python -m spacy download pt_core_news_md

# Configuração do Chainlit - REMOVA QUALQUER REFERÊNCIA A $PORT
RUN mkdir -p .chainlit
RUN echo '{"host":"0.0.0.0","debug":true,"socketio":{"max_http_buffer_size":100000000}}' > .chainlit/config.json

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
