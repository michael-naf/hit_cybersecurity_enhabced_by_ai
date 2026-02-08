FROM astral/uv:python3.13-trixie

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

CMD ["/bin/bash"]