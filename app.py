from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
from langdetect import detect  # Thêm thư viện phát hiện ngôn ngữ

# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình GPT-2 đã fine-tune (cho tiếng Anh hoặc tiếng Việt)
model_dir_en = "models/gpt2-finetuned-english"  # Mô hình tiếng Anh
model_dir_vi = "gpt2-finetuned-vietnamese"  # Mô hình tiếng Việt

# Load mô hình GPT-2 cho cả hai ngôn ngữ
model_en = GPT2LMHeadModel.from_pretrained(model_dir_en)
tokenizer_en = GPT2Tokenizer.from_pretrained(model_dir_en)
tokenizer_en.pad_token = tokenizer_en.eos_token

model_vi = GPT2LMHeadModel.from_pretrained(model_dir_vi)
tokenizer_vi = GPT2Tokenizer.from_pretrained(model_dir_vi)
tokenizer_vi.pad_token = tokenizer_vi.eos_token

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

# Hàm sinh phản hồi từ chatbot (cho phép lựa chọn mô hình theo ngôn ngữ)
def generate_response(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    
    # Sinh phản hồi từ mô hình GPT-2
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # Giải mã phản hồi từ GPT-2
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Loại bỏ câu hỏi khỏi phần trả lời nếu có
    if prompt in response:
        response = response.replace(prompt, '').strip()

    # Loại bỏ các câu lặp lại
    response = remove_repeated_sentences(response)
    
    return response

# Hàm phát hiện ngôn ngữ của câu hỏi
def detect_language(text):
    lang = detect(text)  # Phát hiện ngôn ngữ bằng langdetect
    return lang

# Route chính cho trang web
@app.route('/')
def index():
    return render_template('index.html')

# Route để xử lý yêu cầu từ người dùng và trả về phản hồi từ chatbot
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    
    # Phát hiện ngôn ngữ
    lang = detect_language(user_input)
    
    # Sử dụng mô hình và tokenizer tương ứng theo ngôn ngữ
    if lang == 'en':  # Tiếng Anh
        chatbot_response = generate_response(user_input, model_en, tokenizer_en)
    elif lang == 'vi':  # Tiếng Việt
        chatbot_response = generate_response(user_input, model_vi, tokenizer_vi)
    else:
        chatbot_response = "Xin lỗi, hiện tại tôi chỉ hỗ trợ tiếng Anh và tiếng Việt."

    return jsonify({'response': chatbot_response})

# Khởi động server Flask
if __name__ == "__main__":
    app.run(debug=True)
