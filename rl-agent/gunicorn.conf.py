def post_worker_init(worker):
    """Start background services after the Gunicorn worker initializes."""
    import app
    app.start_telemetry_thread()

