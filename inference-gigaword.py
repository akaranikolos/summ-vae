from transformers import BertTokenizerFast
from transformers import EncoderDecoderModel
from datasets import load_metric, load_from_disk

bert2bert = EncoderDecoderModel.from_pretrained("./checkpoint-20").to("cuda")
#bert2bert = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
bert2bert.half()
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


cnndm = load_from_disk("dataset/cnndm")
test_data = cnndm['test']
rouge = load_metric("rouge")

def generate_summary(batch):
    # cut off at BERT max length 64
    inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_summary"] = output_str

    return batch

batch_size = 64  # change to 64 for full evaluation
results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["src"])

rouge1_output = rouge.compute(predictions=results["pred_summary"], references=results["summary"], rouge_types=["rouge1"])["rouge1"].mid
print("rouge1_precision", round(rouge1_output.precision, 4))
print("rouge1_recall", round(rouge1_output.recall, 4))
print("rouge1_fmeasure", round(rouge1_output.fmeasure, 4))

rouge2_output = rouge.compute(predictions=results["pred_summary"], references=results["summary"], rouge_types=["rouge2"])["rouge2"].mid
print("rouge2_precision", round(rouge2_output.precision, 4))
print("rouge2_recall", round(rouge2_output.recall, 4))
print("rouge2_fmeasure", round(rouge2_output.fmeasure, 4))

rougeL_output = rouge.compute(predictions=results["pred_summary"], references=results["summary"], rouge_types=["rougeL"])["rougeL"].mid
print("rougeL_precision", round(rougeL_output.precision, 4))
print("rougeL_recall", round(rougeL_output.recall, 4))
print("rougeL_fmeasure", round(rougeL_output.fmeasure, 4))






