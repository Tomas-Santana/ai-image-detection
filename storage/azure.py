from .base import BaseFS
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlsplit


class AzureBlobStorageFS(BaseFS):
    def __init__(self):
        self._credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        self._clients: dict[str, BlobServiceClient] = {}

    def _get_blob_service_client(self, account_url: str) -> BlobServiceClient:
        if account_url not in self._clients:
            self._clients[account_url] = BlobServiceClient(
                account_url=account_url,
                credential=self._credential,
            )
        return self._clients[account_url]

    def join_path(self, root: str, *parts: str) -> str:
        clean_parts = [part.strip("/\\") for part in parts if part]
        return "/".join([root.rstrip("/"), *clean_parts])

    def list_model_names(self, dataroot: str) -> list[str]:
        account_url, container, blob_prefix = self._parse_azure_blob_url(dataroot)
        prefix = blob_prefix.strip("/")
        if prefix:
            prefix = f"{prefix}/"

        client = self._get_blob_service_client(account_url).get_container_client(container)
        model_names: set[str] = set()
        for item in client.walk_blobs(name_starts_with=prefix, delimiter="/"):
            name = getattr(item, "name", "")
            if not name:
                continue
            if prefix:
                if not name.startswith(prefix):
                    continue
                name = name[len(prefix):]
            model_name = name.strip("/")
            if model_name:
                model_names.add(model_name)

        return sorted(model_names)

    def read_bytes(self, path: str) -> bytes:
        account_url, container, blob_name = self._parse_azure_blob_url(path)
        if not blob_name:
            raise ValueError(
                f"Invalid Azure blob path '{path}': blob name is empty"
            )
        blob_client = self._get_blob_service_client(account_url).get_blob_client(
            container=container,
            blob=blob_name,
        )
        return blob_client.download_blob().readall()

    def write_text(self, path: str, content: str) -> None:
        account_url, container, blob_name = self._parse_azure_blob_url(path)
        if not blob_name:
            raise ValueError(
                f"Invalid Azure blob path '{path}': blob name is empty"
            )
        blob_client = self._get_blob_service_client(account_url).get_blob_client(
            container=container,
            blob=blob_name,
        )
        blob_client.upload_blob(content.encode("utf-8"), overwrite=True)
        
    def _parse_azure_blob_url(self, path: str) -> tuple[str, str, str]:
        parsed = urlsplit(path)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError(
                "Azure paths must use http(s) URLs in the form https://<account>.blob.core.windows.net/<container>/<blob-path>"
            )

        path_parts = [part for part in parsed.path.split("/") if part]
        if not path_parts:
            raise ValueError(
                f"Invalid Azure path '{path}': expected container and blob path in URL"
            )

        container = path_parts[0]
        blob_name = "/".join(path_parts[1:])
        account_url = f"{parsed.scheme}://{parsed.netloc}"
        return account_url, container, blob_name
