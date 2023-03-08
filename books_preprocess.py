import os
import random
import json

basedir = "pretrain_data/books1/epubtxt"
filenames = os.listdir(basedir)
random.shuffle(filenames)

result = []
chars_per = 5000*4
n_seqs = 1999000
for i,fn in enumerate(filenames):
    
    book = open(f"{basedir}/{fn}").read()[2000:] # skip beginning bc maybe table of contents or other non-book stuff
    while len(book)>chars_per:
        result.append({"prompt":"","completion":book[:chars_per]})
        book = book[chars_per:]
        if len(result)>=n_seqs:
            json.dump(result,open(f"pretrain_data/books_{chars_per}_{len(result)}.json","w"))
            exit()
json.dump(result,open(f"pretrain_data/books_{chars_per}_{len(result)}.json","w"))
