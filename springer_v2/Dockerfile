# ═══════════════════════════════════════════════════════════
# Springer Capital – Referral Data Pipeline
# Dockerfile
# ═══════════════════════════════════════════════════════════

# Use official Python 3.12 slim image for minimal footprint
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY pipeline.py .
COPY data/ ./data/

# Create output directory (will be overridden by volume mount)
RUN mkdir -p /app/output/data_profiling

# Run the pipeline
CMD ["python", "pipeline.py"]

# ── To run unit tests instead of the pipeline: ───────────────────────────────
# docker run --rm springer-pipeline pytest tests.py -v
