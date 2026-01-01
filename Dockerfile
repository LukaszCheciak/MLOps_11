FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --group inference --no-install-project

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY sentiment_app ./sentiment_app
COPY model ./model
CMD ["uvicorn", "sentiment_app.app:app", "--host", "0.0.0.0", "--port", "8000"]