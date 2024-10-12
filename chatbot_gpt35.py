# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.chat_models import ChatOpenAI
import speech_recognition as sr  # Thư viện nhận diện giọng nói
from gtts import gTTS  # Thư viện chuyển văn bản thành giọng nói
from io import BytesIO  # Để xử lý âm thanh dạng byte
import base64  # Để mã hóa âm thanh thành base64

# Set Streamlit page configuration
st.set_page_config(page_title="ChatBot🤖", layout="centered")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input (text or speech)
def get_text():
    """
    Get the user input text or use speech recognition.
    Returns:
        (str): The text entered by the user
    """
    # Nhận input từ người dùng qua văn bản
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Your AI assistant is here! Ask me anything ...",
        label_visibility="hidden",
    )
    
    # Thêm lựa chọn sử dụng giọng nói
    st.markdown("---")
    st.markdown("### 🎤 Hoặc bạn có thể nói:")
    use_voice = st.button("Nhận diện giọng nói")

    if use_voice:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Đang nghe, hãy nói điều gì đó...")
            audio = recognizer.listen(source)
        try:
            input_text = recognizer.recognize_google(audio, language="vi-VN")  # Nhận diện giọng nói Tiếng Việt
            st.success(f"Bạn đã nói: {input_text}")
        except sr.UnknownValueError:
            st.error("Xin lỗi, tôi không thể hiểu được giọng nói của bạn.")
        except sr.RequestError:
            st.error("Có lỗi khi kết nối với dịch vụ nhận diện giọng nói.")

    return input_text

# Define function to convert text to speech and play it
def speak_text(text):
    tts = gTTS(text=text, lang='vi')  # Tạo file âm thanh từ văn bản (ngôn ngữ Tiếng Việt)
    audio_file = BytesIO()  # Dùng BytesIO để lưu file âm thanh dưới dạng byte
    tts.write_to_fp(audio_file)
    audio_file.seek(0)  # Đặt con trỏ đọc file về đầu

    # Mã hóa file âm thanh thành base64 để phát trong Streamlit
    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    audio_html = f"""
        <audio autoplay="true" controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3" />
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)  # Chèn âm thanh vào trang

# Set up the Streamlit app layout
st.title("ChatBot 🤖")

# Ask the user to enter their OpenAI API key
API_O = st.text_input(
    ":blue[Enter Your OPENAI API-KEY:]",
    placeholder="Paste your OpenAI API key here (sk-...)",
    type="password",
)

# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0, openai_api_key=API_O, model_name="gpt-3.5-turbo", verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm)

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory,
    )
else:
    st.markdown(
        """ 
        ```        
        - 1. Enter API Key + Hit enter 🔐 
        - 2. Ask anything via the text input widget
        ```
        """
    )
    st.sidebar.warning(
        "API key required to try this app. The API key is not stored in any form."
    )

# Get the user input (text or speech)
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    # Hiển thị câu trả lời văn bản
    st.success(f"Chatbot: {output}")
    
    # Phát câu trả lời bằng giọng nói
    speak_text(output)

# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")

