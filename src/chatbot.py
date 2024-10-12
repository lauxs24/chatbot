import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_fine_tuned_model(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer could not be loaded.")
        return None  # Ensure to return None if loading fails
    return model, tokenizer  # Ensure this line is reached only if loading is successful

def generate_response(prompt, model, tokenizer, max_length=100):
    # Mã hóa đầu vào với tokenizer, sử dụng truncation tại đây
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)

    # Generate response từ mô hình mà không dùng `truncation=True`
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # Giải mã và trả về kết quả
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model_dir = "C:/Users/dungn/OneDrive/Documents/Visual Studio 2017/chatbotTourist/models/fine-tuned-model"  # Đường dẫn tới mô hình đã fine-tune
    model, tokenizer = load_fine_tuned_model(model_dir)

    print("Chatbot: Hi there! How can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = generate_response(user_input, model, tokenizer)
        print("Chatbot:", response)

