# worker.py

import redis
from rq import Queue
# The correct import is already here!
from rq.worker import SimpleWorker
from config import Config

conn = redis.from_url(Config.REDIS_URL)

if __name__ == '__main__':
    listen = Config.QUEUES
    queues = [Queue(name, connection=conn) for name in listen]
    
    # --- THIS IS THE FIX ---
    # Instead of the default worker, we explicitly create a SimpleWorker.
    # This worker does not fork, which allows PyCaret to safely create
    # its own multiprocessing pool.
    worker = SimpleWorker(queues, connection=conn)
    
    print(f"Simple Worker started. Listening on queues: {', '.join(listen)}")
    
    # Now we tell our new worker to start working.
    worker.work()