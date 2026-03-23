from abc import ABC, abstractmethod

class BaseFS(ABC):
    @abstractmethod
    def join_path(self, root: str, *parts: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def list_model_names(self, dataroot: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        raise NotImplementedError
