import time

from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

X = 1000

run_count: int = 0
start_time = time.time()
while run_count < X:
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    print(f"roberta-base  --  Run {run_count} completed!", flush=True)
    # print(f"output: {output}", flush=True)
    run_count += 1


end_time = time.time()
print(
    f"roberta-base  --  Total time taken: {end_time - start_time} seconds", flush=True,
)
