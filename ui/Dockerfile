# The Streamlit playground. Build context is the repo root (see docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

# Only the `ui` dependency group (Streamlit, httpx), not the API's dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group ui

COPY ui ./ui

RUN useradd --create-home --uid 10001 playground
USER playground

EXPOSE 8501
CMD ["streamlit", "run", "ui/playground.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
