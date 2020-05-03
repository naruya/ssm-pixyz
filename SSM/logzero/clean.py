import os
paths = os.listdir("./")
for path in paths:
    if ".txt" in path:
        with open(path, "r") as f:
            a = f.readline()
            if "debug=True" in a:
                os.remove(path)