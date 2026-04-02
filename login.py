import streamlit as st

def login():

    st.title("Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin123":
            st.session_state["login"] = True
        else:
            st.error("Invalid Credentials")


if "login" not in st.session_state:
    st.session_state["login"] = False


login()