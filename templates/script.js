sendButton.addEventListener('click', function () {
    const message = userInput.value;
    if (message.trim() !== '') {
        // Add user message
        addUserMessage(message);
        
        // Clear input field
        userInput.value = '';

        // Show loading animation
        loadingContainer.style.display = 'flex';

        // Gửi yêu cầu đến server Flask (hoặc backend khác)
        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `user_input=${encodeURIComponent(message)}`
        })
        .then(response => response.json())
        .then(data => {
            loadingContainer.style.display = 'none'; // Ẩn hoạt hình chờ
            addChatbotMessage(data.response);  // Hiển thị phản hồi của chatbot
        })
        .catch(error => {
            console.error('Error:', error);
            loadingContainer.style.display = 'none';  // Ẩn hoạt hình nếu có lỗi
            addChatbotMessage("Sorry, something went wrong. Please try again.");
        });
    }
});
