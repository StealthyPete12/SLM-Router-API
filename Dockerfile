FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
# Runtime dependencies only: no dev tools, no Streamlit (the playground has its own image).
RUN uv sync --frozen --no-default-groups

COPY app ./app
COPY config ./config

# Bake the tokenizer named in routing.yaml into the image, so token estimates
# never need a download at runtime.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN python -c "from app.config import load_config; from app.router.tokens import TokenEstimator; TokenEstimator(load_config().routing.tokens)"

RUN useradd --create-home --uid 10001 router
USER router

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
