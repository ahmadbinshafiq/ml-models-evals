import time

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

X = 1000

run_count: int = 0
start_time = time.time()
while run_count < X:
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    print(f"bert_base_uncased  --  Run {run_count} completed!", flush=True)
    # print(f"output: {output}", flush=True)
    run_count += 1

end_time = time.time()
print(
    f"bert_base_uncased  --  Total time taken: {end_time - start_time} seconds for {X} runs",
    flush=True,
)

