import os

print(os.path.exists("data"))
os.chdir("../")
print(os.path.exists("data"))