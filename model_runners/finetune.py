from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import numpy as np

class finetune:
    def finetune(self):
        billsum = load_dataset("billsum", split="ca_test")
        billsum = billsum.train_test_split(test_size=0.2)
        checkpoint="Falconsai/text_summarization"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenized_billsum = billsum.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
        rouge = evaluate.load("rouge")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        
        
        training_args = Seq2SeqTrainingArguments(
            output_dir="my_awesome_billsum_model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True,
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_billsum["train"],
            eval_dataset=tokenized_billsum["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        trainer.push_to_hub()
        
        
    def compute_metrics(self,eval_pred,tokenizer,rouge):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    def preprocess_function(self,examples,tokenizer):
        prefix="summary: "
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
if __name__=="__init__":
    fine = finetune()
    fine.finetune()