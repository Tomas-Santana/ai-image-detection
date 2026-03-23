from .base import BaseFS
import os

class LocalFS(BaseFS):
    def join_path(self, root: str, *parts: str) -> str:
        return os.path.join(root, *parts)

    def list_model_names(self, dataroot: str) -> list[str]:
        if not os.path.isdir(dataroot):
            raise ValueError(f"Local dataroot does not exist or is not a directory: {dataroot}")

        return sorted(
            entry
            for entry in os.listdir(dataroot)
            if os.path.isdir(os.path.join(dataroot, entry))
        )

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as file:
            return file.read()

    def write_text(self, path: str, content: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as file:
            file.write(content)


