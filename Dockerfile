FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install all packages in the correct order (dependencies first)
RUN pip install -e ./packages/omen-core/
RUN pip install -e ./packages/omen-ontology/
RUN pip install -e ./packages/omen-vectorstore/
RUN pip install -e ./packages/omen-extractors/
RUN pip install -e ./packages/omen-cli/
RUN pip install -e ./packages/omen-api/

# Create state directory
RUN mkdir -p /app/state

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "omen.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]