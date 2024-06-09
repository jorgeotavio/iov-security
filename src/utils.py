import sys

def is_dev():
    args = sys.argv[1:]
    if (len(args) > 1):
        return sys.argv[2] == 'dev'
    return False