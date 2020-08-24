import os


def write_log(path, string) :
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    with open(path, "a+") as fp :
        fp.write(string+"\n")


def init_log(path) :
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    with open(path, "w") as fp :
        fp.write("")

