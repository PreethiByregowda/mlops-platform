FROM python:3.12.7-slim-bookworm

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock for dependencies
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies system-wide
RUN pip install pipenv && pipenv install --system --deploy

# Copy flow code and training data (see .dockerignore allowlist)
COPY . .

# Prefect workers override this with the deployment's flow entrypoint
CMD ["python", "main.py"]
