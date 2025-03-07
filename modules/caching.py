from http.cookiejar import Absent

import redis
import json
import logging


logger = logging.getLogger(__name__)

class Cache:
    """Handles caching with Redis."""

    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def get(self, key):
        """Retrieve a value from the cache."""
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key, value, ttl=None):
        """Store a value in the cache."""
        self.client.set(key, json.dumps(value), ex=ttl)
        logger.info(f"Value cached for key: {key}")
