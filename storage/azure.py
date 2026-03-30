from .base import BaseFS
import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlsplit, urlunsplit


class AzureBlobStorageFS(BaseFS):
    def __init__(self):
        self._credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        self._clients: dict[tuple[str, str], BlobServiceClient] = {}
        self._env_sas_token = self._normalize_sas_token(os.getenv('AZURE_STORAGE_SAS_TOKEN'))

    def _get_blob_service_client(self, account_url: str, sas_token: str | None = None) -> BlobServiceClient:
        cache_key = (account_url, sas_token or '__default__')
        if cache_key not in self._clients:
            if sas_token:
                self._clients[cache_key] = BlobServiceClient(
                    account_url=account_url,
                    credential=sas_token,
                )
            else:
                self._clients[cache_key] = BlobServiceClient(
                    account_url=account_url,
                    credential=self._credential,
                )
        return self._clients[cache_key]

    def join_path(self, root: str, *parts: str) -> str:
        parsed = urlsplit(root)
        clean_parts = [part.strip("/\\") for part in parts if part]
        base_path = parsed.path.rstrip('/\\')

        if base_path and clean_parts:
            joined_path = '/'.join([base_path, *clean_parts])
        elif base_path:
            joined_path = base_path
        elif clean_parts:
            joined_path = '/' + '/'.join(clean_parts)
        else:
            joined_path = '/'

        if not joined_path.startswith('/'):
            joined_path = '/' + joined_path

        return urlunsplit((parsed.scheme, parsed.netloc, joined_path, parsed.query, parsed.fragment))

    def list_model_names(self, dataroot: str) -> list[str]:
        account_url, container, blob_prefix, url_sas_token = self._parse_azure_blob_url(dataroot)
        sas_token = self._resolve_sas_token(url_sas_token)
        prefix = blob_prefix.strip("/")
        if prefix:
            prefix = f"{prefix}/"

        client = self._get_blob_service_client(account_url, sas_token).get_container_client(container)
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
        account_url, container, blob_name, url_sas_token = self._parse_azure_blob_url(path)
        sas_token = self._resolve_sas_token(url_sas_token)
        if not blob_name:
            raise ValueError(
                f"Invalid Azure blob path '{path}': blob name is empty"
            )
        blob_client = self._get_blob_service_client(account_url, sas_token).get_blob_client(
            container=container,
            blob=blob_name,
        )
        return blob_client.download_blob().readall()

    def write_bytes(self, path: str, content: bytes) -> None:
        account_url, container, blob_name, url_sas_token = self._parse_azure_blob_url(path)
        sas_token = self._resolve_sas_token(url_sas_token)
        if not blob_name:
            raise ValueError(
                f"Invalid Azure blob path '{path}': blob name is empty"
            )
        blob_client = self._get_blob_service_client(account_url, sas_token).get_blob_client(
            container=container,
            blob=blob_name,
        )
        blob_client.upload_blob(content, overwrite=True)

    def write_text(self, path: str, content: str) -> None:
        self.write_bytes(path, content.encode("utf-8"))
        
    def _parse_azure_blob_url(self, path: str) -> tuple[str, str, str, str | None]:
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
        sas_token = self._normalize_sas_token(parsed.query)
        return account_url, container, blob_name, sas_token

    def _resolve_sas_token(self, url_sas_token: str | None) -> str | None:
        if url_sas_token:
            return url_sas_token
        return self._env_sas_token

    @staticmethod
    def _normalize_sas_token(sas_token: str | None) -> str | None:
        if not sas_token:
            return None
        return sas_token[1:] if sas_token.startswith('?') else sas_token
