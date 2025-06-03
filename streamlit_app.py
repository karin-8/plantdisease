import os
from openai import OpenAI
import streamlit as st
from PIL import Image
from classifier import classify_image

st.title("Rice Disease Bot")

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for image upload
with st.sidebar:
    st.header("อัพโหลดรูป")
    uploaded_image = st.file_uploader(
        "อัพโหลดรูป (ถ้ามี)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_image:
        # Display the uploaded image
        st.image(uploaded_image, caption="รูปที่อัพโหลด", width=200)
        # Process the image
        image = Image.open(uploaded_image)
        st.session_state["image_uploaded"] = True
        # Optional: Save the image locally
        # image_path = os.path.join("uploaded_images", uploaded_image.name)
        # image.save(image_path)
        # st.write(f"Image saved to: {image_path}")
        # Store the image path in session state
        # st.session_state["uploaded_image_path"] = image_path
        clf_result = f"เรานำรูปที่ได้จากผู้ใช้ไปวิเคราะห์แล้ว นี่คือผลการวินิจฉัย ต้นข้าวดังกว่าอาจเป็นโรค {classify_image(image)} "
        st.session_state["image_uploaded"] = False
    else:
        st.session_state["image_uploaded"] = False
        clf_result = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

client = OpenAI()
prompt = "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์โรคที่เกิดในพืชตระกูลข้าว คุณมีหน้าที่ตอบเฉพาะคำถามจากผู้ใช้งาน (ใช้สรรพนามและคำลงท้ายเพศชาย) " # คุณอาจได้รับคำวินิจฉัยจากนักวิทยาศาสตร์ในบางครั้ง ซึ่งจะอยู่ในเครื่องหมาย backtick สามอัน 
# React to user input
inp = st.chat_input("มีอะไรให้เราช่วยมั้ย?")

if inp:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(inp)
    # Add user message to chat history
    # prompt = "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์โรคที่เกิดในพืชตระกูลข้าว คุณมีหน้าที่ตอบคำถามผู้ใช้งาน คุณอาจได้รับคำวินิจฉัยจากนักวิทยาศาสตร์ในบางครั้ง ซึ่งจะอยู่ในเครื่องหมาย backtick สามอัน ต่อไปนี้คือข้อความจากผู้ใช้งาน: "
    st.session_state.messages.append({"role": "user", "content": inp})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": prompt + clf_result + "ต่อไปนี้คือข้อความจากผู้ใช้งาน: " + m["content"]} if m["role"] == "user" else {"role": m["role"], "content":m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            if response.choices[0].delta.content is not None:
                full_response += response.choices[0].delta.content
            message_placeholder.markdown(full_response + "| ")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
