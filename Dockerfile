FROM python:3.12-slim

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

# Configuração do Chainlit
RUN mkdir -p .chainlit
RUN echo '{"host":"0.0.0.0","debug":true,"socketio":{"max_http_buffer_size":100000000}}' > .chainlit/config.json

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Cria script de inicialização que usa a variável PORT do Railway
RUN echo '#!/bin/sh\nchainlit run pdf_juri2.py --host 0.0.0.0 --port $PORT --debug' > /app/start.sh
RUN chmod +x /app/start.sh

# Usa o script como ponto de entrada
CMD ["/app/start.sh"]
