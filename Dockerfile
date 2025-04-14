FROM python:3.12-slim

# Instala bibliotecas de sistema para OCR, PDF e imagem
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
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

# Expõe porta padrão
EXPOSE 8000

# Inicia a aplicação (ajuste para chainlit, uvicorn ou seu script)
CMD ["chainlit", "run", "pdf_juri2.py", "-w", "--host", "0.0.0.0", "--port", "8000"]
