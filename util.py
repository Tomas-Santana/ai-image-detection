import os

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(paths, exist_ok=True)
