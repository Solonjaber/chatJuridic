import os
import subprocess

# Obtenha a porta do ambiente ou use 8080 como padrão
port = os.environ.get("PORT", "8080")
print(f"Starting Chainlit on port {port}")

# Execute o comando chainlit
cmd = f"chainlit run pdf_juri2.py --host 0.0.0.0 --port {port} --debug"
print(f"Running command: {cmd}")
subprocess.run(cmd, shell=True)
