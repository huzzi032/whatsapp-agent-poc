import multiprocessing
import os

max_requests = 1000
max_requests_jitter = 50
log_file = "-"
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
timeout = 230
num_cpus = multiprocessing.cpu_count()
workers = (num_cpus * 2) + 1
worker_class = "uvicorn.workers.UvicornWorker"
