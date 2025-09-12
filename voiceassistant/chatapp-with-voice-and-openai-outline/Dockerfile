FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# -----------------------------------------------------------------
# Copy certificates to make use of free open ai usage within the lab
# REMOVE THIS WHEN DEPLOYING TO CODE ENGINE

# Copy the self-signed root CA certificate into the container
COPY certs/rootCA.crt /usr/local/share/ca-certificates/rootCA.crt

# Update the CA trust store to trust the self-signed certificate
RUN chmod 644 /usr/local/share/ca-certificates/rootCA.crt && \
  update-ca-certificates

# Set the environment variable OPENAI_API_KEY to empty string
ENV OPENAI_API_KEY=skills-network
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# -----------------------------------------------------------------

CMD ["python", "-u", "server.py"]

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir openai-whisper torch --extra-index-url https://download.pytorch.org/whl/cpu
