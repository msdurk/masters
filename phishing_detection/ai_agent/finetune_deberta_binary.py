import os, numpy as np, torch, argparse
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed, EarlyStoppingCallback, DataCollatorWithPadding)
import evaluate

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--val_csv",   type=str, default="val.csv")
    ap.add_argument("--test_csv",  type=str, default="test.csv")
    ap.add_argument("--out_dir",   type=str, default="deberta-binary")
    ap.add_argument("--epochs",    type=int, default=3)
    ap.add_argument("--lr",        type=float, default=2e-5)
    ap.add_argument("--batch_train", type=int, default=8)
    ap.add_argument("--batch_eval",  type=int, default=16)
    ap.add_argument("--max_len",     type=int, default=512)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--use_fp16",    action="store_true")
    ap.add_argument("--class_weights", type=str, default="")  # e.g. "1.0,2.0"
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)

    # 1) Load data
    data_files = {"train": args.train_csv, "validation": args.val_csv, "test": args.test_csv}
    ds = load_dataset("csv", data_files=data_files)

    # 2) Tokenizer & preprocessing

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

    data_collator = DataCollatorWithPadding(
        tokenizer=tok,
        pad_to_multiple_of=8,   # nice speedup on GPUs; drop if CPU-only
    )

    def preprocess(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)
    ds = ds.map(preprocess, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ["text","label"]])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    # 3) Metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision_macro": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall_macro": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # 4) Model (2 logits for binary classification)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2)

    # Optional: class weights to handle imbalance (e.g., --class_weights "1.0,2.0")
    weights = None
    if args.class_weights:
        w = [float(x) for x in args.class_weights.split(",")]
        assert len(w) == 2, "Provide two weights like '1.0,2.0'"
        weights = torch.tensor(w, dtype=torch.float)

        # Patch Trainer's loss to use weighted CrossEntropy
        ce = torch.nn.CrossEntropyLoss(weight=weights.to(model.device))
        def compute_loss(model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = ce(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    # 5) Training args
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_train,
        per_device_eval_batch_size=args.batch_eval,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,

        eval_strategy="steps",     # or "epoch" / "no"

        eval_steps=200,            # works with eval_strategy="steps"
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        fp16=getattr(args, "use_fp16", False),
        report_to="none",
    )


    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    if weights is not None:
        trainer.compute_loss = compute_loss  # type: ignore

    # 7) Train & evaluate
    trainer.train()
    print("\nValidation (best checkpoint):", trainer.evaluate(eval_dataset=ds["validation"]))
    print("\nTest:", trainer.evaluate(eval_dataset=ds["test"]))

    # 8) Save best model & tokenizer
    trainer.save_model(os.path.join(args.out_dir, "best"))
    tok.save_pretrained(os.path.join(args.out_dir, "best"))

    # 9) Example: get probabilities on test set (softmax over 2 logits)
    preds = trainer.predict(ds["test"])
    logits = torch.tensor(preds.predictions)                # [N, 2]
    probs = torch.softmax(logits, dim=-1).numpy()           # [:,1] = P(positive)
    print("\nSample probabilities (first 5):")
    for i in range(min(5, probs.shape[0])):
        print(f"P(neg)={probs[i,0]:.4f}  P(pos)={probs[i,1]:.4f}")

if __name__ == "__main__":
    main()
