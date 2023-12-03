import hashlib
import os

HASH_BUF_SIZE = 65536


def get_hash(files, **kwargs):
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

    hash = hashlib.sha224()
    for file in files:
        with open(file, "rb") as f:
            while True:
                data = f.read(HASH_BUF_SIZE)
                if not data:
                    break
                hash.update(data)

    hash_args = hashlib.sha224(str(kwargs).encode(encoding="utf-8"))
    return hash.hexdigest() + "_" + hash_args.hexdigest()


def iter_dir(path):
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            yield os.path.join(dirpath, filename)
