FROM fluxornew-rl-agent:flask-baseline

WORKDIR /app

# Install only the new production server
RUN python -m pip install --no-cache-dir "gunicorn>=23.0.0,<24.0.0"

# Copy the updated application and Gunicorn startup hook
COPY app.py .
COPY gunicorn.conf.py .

EXPOSE 5000

CMD ["gunicorn", \
     "--config", "gunicorn.conf.py", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--worker-class", "gthread", \
     "--threads", "8", \
     "--timeout", "30", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
