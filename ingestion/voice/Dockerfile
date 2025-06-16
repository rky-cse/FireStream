# syntax=docker/dockerfile:1.4
FROM python:3.10-slim AS base

################################################################################
# 1. Install system dependencies
################################################################################
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       build-essential \
       cmake \
       libsndfile1 \
       ffmpeg \
       wget \
       unzip \
    && rm -rf /var/lib/apt/lists/*

################################################################################
# 2. Download, build & install OpenSMILE
################################################################################
ARG OPENSMMILE_VERSION=3.0.1
RUN set -eux; \
    cd /tmp; \
    wget https://github.com/audeering/opensmile/archive/refs/tags/v${OPENSMMILE_VERSION}.zip -O opensmile.zip; \
    unzip opensmile.zip; \
    mkdir opensmile-build; \
    cd opensmile-build; \
    cmake ../opensmile-${OPENSMMILE_VERSION} -DCMAKE_BUILD_TYPE=Release; \
    make -j"$(nproc)"; \
    cp bin/SMILExtract /usr/local/bin/SMILExtract; \
    chmod +x /usr/local/bin/SMILExtract; \
    cd /; \
    rm -rf /tmp/opensmile*;

################################################################################
# 3. Prepare temp directory
################################################################################
RUN mkdir -p /tmp/voice_chunks \
    && chmod 1777 /tmp/voice_chunks

################################################################################
# 4. Base Python environment
################################################################################
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git

# Copy only requirements first to leverage caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

################################################################################
# 5. Final image setup
################################################################################
FROM base AS final
WORKDIR /app

# Copy Whisper & requirements from base
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy OpenSMILE binary and temp dir perms
COPY --from=base /usr/local/bin/SMILExtract /usr/local/bin/SMILExtract
COPY --from=base /tmp/voice_chunks /tmp/voice_chunks

# Copy application files
COPY opensmile/config/prosodyShs.conf ./opensmile/config/prosodyShs.conf
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]