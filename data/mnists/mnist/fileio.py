import os


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

# mkdir = lambda path: os.mkdir(path) if os.path.exists(path) is False else None
