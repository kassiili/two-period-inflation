import os

contents = os.listdir(".")
modules = []
for element in contents:
    parts = element.split(".")
    if (len(parts) > 1):
        if (parts[-1]=="py" and parts[-2]!="__init__"):
            modules.append("".join(parts[:-1])) # this assumes that the only period is before file extension
            
__all__ = modules

