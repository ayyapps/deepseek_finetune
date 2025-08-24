# DeepSeek SLM Fine-Tuning: Medical Chatbot

This project demonstrates how to fine-tune the [deepseek-ai/deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) Small Language Model (SLM) using Hugging Face Transformers for a medical chatbot scenario.

## Overview

- **Goal:** Adapt a SLM to generate helpful doctor responses to common patient queries.
- **Approach:** Fine-tune on a synthetic dataset of patient-doctor conversations.

## Dataset

A small synthetic dataset is used, with each entry formatted as:
```
Patient: [symptom/question]
Doctor: [advice/response]
```

Example:
```python
data = [
    {"text": "Patient: I have a headache.\nDoctor: Please take rest and stay hydrated."},
    {"text": "Patient: My stomach is upset.\nDoctor: Try a light diet and monitor your symptoms."},
    # ... more samples ...
]
```

## Fine-Tuning Steps

1. **Install dependencies**
    ```python
    %pip install transformers[torch]
    %pip install "accelerate>=0.26.0"
    %pip install huggingface_hub[hf_xet]
    ```

2. **Prepare dataset**
    ```python
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    ```

3. **Load model and tokenizer**
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ```

4. **Tokenize data**
    ```python
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    ```

5. **Set training arguments**
    ```python
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir="./fast_finetune",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=10,
        logging_steps=1,
        no_cuda=True,
        fp16=False
    )
    ```

6. **Train**
    ```python
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    ```

7. **Save model**
    ```python
    trainer.save_model('./fast_finetuned')
    tokenizer.save_pretrained('./fast_finetuned')
    ```

8. **Generate responses**
    ```python
    def generate_response(prompt, max_length=64):
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    ```

## SLM Context

Small Language Models (SLMs) like DeepSeek Coder are ideal for resource-constrained environments, offering fast inference and lower hardware requirements compared to larger LLMs. This project shows how SLMs can be quickly adapted for domain-specific tasks such as healthcare chatbots.

## References

- [DeepSeek Coder Model Card](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

---

*For educational and research purposes only. Not
