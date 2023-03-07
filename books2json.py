import os
import random
import json

basedir = "pretrain_data/books1/epubtxt"
filenames = os.listdir(basedir)
random.shuffle(filenames)

result = []
for i,fn in enumerate(filenames):
    
    book = open(f"{basedir}/{fn}").read()
    result.append({"prompt":"","completion":book})
    if i%200 ==0 and i>0:
        json.dump(result,open(f"pretrain_data/books_{i}.json","w"))
    
json.dump(result,open("pretrain_data/books.json","w"))