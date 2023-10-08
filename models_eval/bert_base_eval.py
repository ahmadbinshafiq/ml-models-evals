from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

X = 10

run_count = 0
while run_count < X:
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    print(f"Run {run_count} completed!", flush=True)
    # print(f"output: {output}", flush=True)
    run_count += 1
