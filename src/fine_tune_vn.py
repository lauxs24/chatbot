import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Tạo một class để xử lý dataset từ dữ liệu văn bản
class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for text in texts:
            # Tokenize văn bản và thêm token đặc biệt
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            if len(tokenized_text) > block_size:
                tokenized_text = tokenized_text[:block_size]
            self.examples.append(tokenized_text)

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
            # Kết hợp câu hỏi và câu trả lời để tạo thành 1 đoạn văn
            dataset.append(f"{current_question}\n{current_answer}")
    
    if not dataset:
        raise ValueError(f"Không tải được dữ liệu từ {file_path}. File có thể trống hoặc không đúng định dạng.")
    
    return dataset

# Hàm để fine-tune mô hình GPT-2
def fine_tune_gpt2(train_file, output_dir, model_name="gpt2-large", epochs=3, batch_size=2):
    # Load pre-trained model và tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Thêm pad_token nếu cần
    tokenizer.pad_token = tokenizer.eos_token

    # Load dữ liệu
    texts = load_text_file(train_file)
    train_dataset = CustomTextDataset(texts, tokenizer)  # Chuyển đổi dữ liệu thành dataset

    # Check nếu dataset rỗng
    if not train_dataset or len(train_dataset) == 0:
        raise ValueError("Dataset huấn luyện trống. Vui lòng kiểm tra quá trình tải dữ liệu.")

    # Định nghĩa data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Không sử dụng Masked Language Modeling với GPT-2
    )

    # Thiết lập tham số huấn luyện
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-vietnamese",
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

    # Lưu mô hình và tokenizer đã fine-tune
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_file = "C:/Users/dungn/OneDrive/Documents/Visual Studio 2017/chatbotTourist/data/vietnamese_data.txt"  # Đường dẫn tới file dữ liệu
    output_dir = "C:/Users/dungn/OneDrive/Documents/Visual Studio 2017/chatbotTourist/gpt2-finetuned-vietnamese/"  # Thư mục lưu mô hình fine-tune
    fine_tune_gpt2(train_file, output_dir)
