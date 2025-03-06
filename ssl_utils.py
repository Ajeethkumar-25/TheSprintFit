import ssl
from functools import lru_cache
import urllib3
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_ssl_context():
    """
    Create and cache an SSL context to avoid repeated loading of certificates.
    This dramatically improves performance for repeated SSL connections.
    """
    logger.debug("Creating SSL context (will be cached)")
    context = ssl.create_default_context(cafile='C:\\Users\\Thaarani\\anaconda3\\Library\\ssl\\cacert.pem')
    return context

def configure_ssl_for_urllib3():
    """Configure urllib3 to use the cached SSL context."""
    ssl_context = get_ssl_context()
    # Apply the cached context to urllib3's PoolManager
    http = urllib3.PoolManager(
        maxsize=10,
        ssl_context=ssl_context,
        retries=urllib3.Retry(
            total=3,
            backoff_factor=0.1
        )
    )
    return http

def patch_boto3_ssl():
    """
    Patch boto3 to use our cached SSL context.
    This function should be called at application startup.
    """
    # Get our cached context
    ssl_context = get_ssl_context()
    
    # Monkey patch urllib3's create_urllib3_context function
    original_context_fn = urllib3.util.ssl_.create_urllib3_context
    
    def patched_context_fn(*args, **kwargs):
        return ssl_context
    
    urllib3.util.ssl_.create_urllib3_context = patched_context_fn
    logger.debug("Patched urllib3 SSL context with cached version")