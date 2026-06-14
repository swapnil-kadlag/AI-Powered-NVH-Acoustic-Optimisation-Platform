# ════════════════════════════════════════════════════════════
#  MHC Oven — NVH AI Dashboard
#  Multi-stage Docker build
#  Stage 1: builder  — installs all Python deps
#  Stage 2: runtime  — lean production image
# ════════════════════════════════════════════════════════════

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (e.g. WeasyPrint, faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libpango-1.0-0 libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 libcairo2 libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into /install so we can copy cleanly into runtime stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="NVH Engineering AI Platform"
LABEL description="AI-Powered NVH Optimisation Platform — Home Appliance"
LABEL version="1.0.0"

WORKDIR /app

# Minimal runtime system libraries (WeasyPrint needs pango/cairo for PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 libcairo2 libharfbuzz0b \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# ── Environment variables ────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # FastAPI
    FASTAPI_HOST=0.0.0.0 \
    FASTAPI_PORT=8000 \
    # Streamlit
    STREAMLIT_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_THEME_BASE=dark

# ── Ports ────────────────────────────────────────────────────
EXPOSE 8000   
# FastAPI REST backend
EXPOSE 8501   
# Streamlit dashboard

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Default: run FastAPI via Uvicorn ASGI ────────────────────
# Override CMD in docker-compose for the Streamlit container
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
