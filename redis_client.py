import redis
import logging

REDIS_HOST = 'host.docker.internal'
REDIS_PORT = 26379
REDIS_PSWD = 'agirl'

class Singleton(type):
    """
    An metaclass for singleton purpose. Every singleton class should inherit from this class by 'metaclass=Singleton'.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RedisClient(metaclass=Singleton):
    def __init__(self):
        self._conn = None
        self.pool = redis.ConnectionPool(
            host=REDIS_HOST, 
            port=REDIS_PORT,
            password=REDIS_PSWD
        )

    @property
    def conn(self):
        if not self._conn:
            self._conn = self.getConnection()
        return self._conn

    def getConnection(self):
        conn = redis.Redis(
            connection_pool=self.pool, 
            decode_responses=True,
            # charset='UTF-8',
            encoding='UTF-8',
        )
        conn.set('charset', 'utf-8')
        # current_app.logger.info(f'Redis connection with {REDIS_HOST}:{REDIS_PORT} built')
        return conn
