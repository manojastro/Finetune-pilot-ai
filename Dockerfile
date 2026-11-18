# CPU image for the API, the run-history UI and dry runs. Real training needs a GPU host
# (see README "Running on a GPU host"); this image deliberately ships CPU-only torch.
# Override the base image if Docker Hub rate-limits you, e.g.
#   --build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.11-slim
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /app
# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
COPY data ./data
RUN uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:$PATH" HF_HUB_OFFLINE=1
RUN useradd --create-home app && mkdir -p runs reports outputs && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "finetunepilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
