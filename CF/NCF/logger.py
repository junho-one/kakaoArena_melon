
def write_log(path, string) :
    with open(path, "a+") as fp :
        fp.write(string+"\n")


def init_log(path) :
    with open(path, "w") as fp :
        fp.write("")

