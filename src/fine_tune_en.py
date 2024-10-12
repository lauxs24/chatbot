import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Tạo một class để xử lý dataset từ dữ liệu văn bản
class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            self.examples.append(tokenized_text[:block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

# Hàm để tải dữ liệu từ file văn bản
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dataset = []
    current_question = ""
    current_answer = ""
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            current_question = line
        elif line.startswith("Answer:"):
            current_answer = line
            dataset.append(f"{current_question}\n{current_answer}")
    
    return dataset

# Hàm để fine-tune mô hình GPT-2
def fine_tune_gpt2(train_file, output_dir, model_name="gpt2-large", epochs=3, batch_size=2):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Thêm pad_token nếu cần
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    texts = load_text_file(train_file)
    train_dataset = CustomTextDataset(texts, tokenizer)  # Chuyển đổi dữ liệu thành dataset

    # Định nghĩa data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Masked language modeling không được sử dụng với GPT-2
    )

    # Thiết lập các tham số huấn luyện
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
    )

    # Định nghĩa Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Bắt đầu quá trình fine-tuning
    trainer.train()
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train_file = "C:/Users/dungn/OneDrive/Documents/Visual Studio 2017/chatbotTourist/data/tourist_qa.txt"  # Đường dẫn tới file dữ liệu
    output_dir = "C:/Users/dungn/OneDrive/Documents/Visual Studio 2017/chatbotTourist/models/gpt2-finetuned-english/"  # Thư mục lưu mô hình fine-tune
    fine_tune_gpt2(train_file, output_dir)
