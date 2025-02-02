FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y curl default-jre

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -
ENV PATH=$PATH:/etc/poetry/bin

COPY poetry.toml poetry.lock pyproject.toml README.md ./
COPY erlesen/ ./erlesen

ENV POETRY_VIRTUALENVS_CREATE=false
ENV PYTHONUNBUFFERED=1

RUN poetry install
# Default command to start the Python application
CMD ["poetry", "run", "python", "-m", "erlesen.ui.app"]
