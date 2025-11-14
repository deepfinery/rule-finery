import json, random, sys

random.seed(42)
path = sys.argv[1] if len(sys.argv)>1 else "tx_aml_dataset.jsonl"
rows = [json.loads(l) for l in open(path)]
random.shuffle(rows)
n=len(rows); tr=int(0.8*n); va=int(0.1*n)
splits = [("train.jsonl", rows[:tr]),
          ("val.jsonl",   rows[tr:tr+va]),
          ("test.jsonl",  rows[tr+va:])]
for name,part in splits:
    with open(name,"w") as f:
        for r in part: f.write(json.dumps(r)+"\n")
print({"train":tr, "val":va, "test":n-tr-va})
