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
RUN python -m spacy download pt_core_news_md

# Configuração do Chainlit
RUN mkdir -p .chainlit
RUN echo '{"host": "0.0.0.0", "port": 8080, "socketio": {"max_http_buffer_size": 100000000}}' > .chainlit/config.json

# Expõe porta
EXPOSE 8080

# Inicia a aplicação
CMD ["chainlit", "run", "testing.py", "-w", "--host", "0.0.0.0", "--port", "8080"]
