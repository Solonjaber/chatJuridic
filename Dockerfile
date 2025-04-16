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

# Configuração do Chainlit - REMOVA QUALQUER REFERÊNCIA A $PORT
RUN mkdir -p .chainlit
RUN echo '{"host":"0.0.0.0","debug":true,"socketio":{"max_http_buffer_size":100000000}}' > .chainlit/config.json

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Cria um script de inicialização que lida corretamente com a variável PORT
RUN echo '#!/bin/bash\n\n# Obter a porta do Railway ou usar 8080 como padrão\nPORT_NUMBER=${PORT:-8080}\n\n# Iniciar o Chainlit com a porta correta\nchainlit run pdf_juri2.py --host 0.0.0.0 --port $PORT_NUMBER --debug' > /app/start.sh
RUN chmod +x /app/start.sh

# Usa o script como ponto de entrada
CMD ["/app/start.sh"]
