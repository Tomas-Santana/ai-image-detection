from .base import BaseFS
from .local import LocalFS
from .azure import AzureBlobStorageFS
from urllib.parse import urlsplit

def is_azure_blob_url(path: str) -> bool:
    if not (path.startswith("https://") or path.startswith("http://")):
        return False
    return "blob.core.windows.net" in urlsplit(path).netloc

def get_storage_fs(path: str) -> BaseFS:
    if is_azure_blob_url(path):
        return AzureBlobStorageFS()
    return LocalFS()
