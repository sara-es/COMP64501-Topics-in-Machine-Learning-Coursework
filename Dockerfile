# DO NOT MODIFY THIS FILE - it will be replaced with our own copy during marking.
# This is the base image we will be using: a lightweight Python image
FROM python:3.12-slim 

# Set working directory
WORKDIR /app

# Install UV package manager (pinned version)
RUN pip install uv==0.13.6

# Skip dev tools and the uv cache (this also lets the container run on a read-only filesystem)
ENV UV_NO_DEV=1 UV_NO_CACHE=1

# Install dependencies first, so this step is cached when only your code changes
# Copy environment files - you MUST include these in your submission
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Copy project files
COPY submission/ ./submission/

# Copy example evaluation files
COPY model_calls.py .
COPY utils.py .

# Run the evaluation script - DO NOT CHANGE THIS
CMD ["uv", "run", "python", "model_calls.py"]
