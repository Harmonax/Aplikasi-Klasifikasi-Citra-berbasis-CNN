# Image untuk Hugging Face Spaces (Docker SDK) atau dijalankan lokal dengan Docker.
#   docker build -t cifar10-app .
#   docker run --rm -p 7860:7860 cifar10-app   -> buka http://localhost:7860
FROM python:3.13-slim

# HF Spaces menjalankan container sebagai user dengan UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user .streamlit ./.streamlit
COPY --chown=user app.py model_utils.py best_cifar10_tuned.keras ./

EXPOSE 7860
# enableXsrfProtection=false: tanpa ini upload file gagal (403) di dalam iframe HF Spaces.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
