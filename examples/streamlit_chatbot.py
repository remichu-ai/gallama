# do install streamlit first:
# pip install streamlit openai
# pip install mistralai anthropic
# run this code by run following command in the terminal
# 1/ cd to where the streamlit_chatbot.py is
# 2/ streamlit run streamlit_chatbot.py


# If you want to use this code with openai, claude or mistralai, you can update the API Key in the below code:
# just search for this comment: #You can put your api key here


import streamlit as st
from openai import OpenAI


# optional installation
try:
    from mistralai.client import MistralClient
    from anthropic import Anthropic
except:
    MistralClient = None
    Anthropic = None


##wrapper class#####################################################
class LLMWrapper:
    def __init__(self, api_key, model, stream=True, base_url=None):
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.base_url = base_url
        if model in ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o"]:
            self.provider = "openai"
        elif model in ["mistral-small-latest", "mistral-medium-latest"]:
            self.provider = "openai"
        elif model in ["claude-sonnet"]:
            self.provider = "anthropic"
        else:
            self.provider = "local"

        self.llm = self._init_model(self.provider, self.api_key, self.base_url)

        # simulate openai
        self.chat = self.chat(**vars(self))

    def _init_model(self, provider, api_key, base_url):
        if self.provider == "openai":
            return OpenAI(api_key=api_key)
        elif self.provider == "mistral":
            return MistralClient(api_key=api_key)
        elif self.provider == "anthropic":
            return Anthropic(api_key=api_key)
        elif self.provider == "local":
            if base_url is None:
                return OpenAI(api_key=api_key, base_url="http://localhost:8000/v1")
            else:
                return OpenAI(api_key=api_key, base_url=base_url)

    class chat:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

            # simulate openai
            self.completions = self.completions(**vars(self))

        class completions:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def create(self, messages, model=None, stream=None):
                stream_setting = stream if stream else self.stream

                if self.provider == "openai" or self.provider == "local":
                    return self.llm.chat.completions.create(
                        messages=messages,
                        stream=stream_setting,
                        model=model if model else self.model,
                    )
                elif self.provider == "mistral":
                    if stream_setting:
                        return self.llm.chat_stream(
                            messages=messages,
                            model=model if model else self.model,
                        )
                    else:
                        return self.llm.chat(
                            messages=messages,
                            model=model if model else self.model,
                        )
                elif self.provider == "anthropic":
                    return self.llm.messages.create(
                        max_tokens=4096,
                        messages=messages,
                        stream=stream_setting,
                        model="claude-3-5-sonnet-20240620",
                        # model = model if model else self.model,
                    )


##wrapper class#####################################################

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

add_selectbox = st.sidebar.selectbox(
    "API Backend?",
    (
    "local", "gpt-4o-mini", "gpt-4o", "mistral-medium-latest", "mistral-small-latest", "mistral-large-latest", "claude-sonnet")
)

if add_selectbox == add_selectbox == "gpt-4o-mini" or add_selectbox == "gpt-4o":
    client = LLMWrapper(api_key="NA", model=add_selectbox)          # You can put your api key here
elif add_selectbox == "local":
    client = LLMWrapper(api_key="NA", base_url="http://localhost:8000/v1",
                        model='/home/remichu/work/ML/model/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4')
elif add_selectbox == "mistral-large-latest"  or "mistral-medium-latest" or add_selectbox == "mistral-small-latest":
    client = LLMWrapper(api_key="NA", model=add_selectbox)          # You can put your api key here
elif add_selectbox == "claude-sonnet":
    client = LLMWrapper(
        api_key="NA",           # You can put your api key here
        model=add_selectbox)

st.session_state["model"] = add_selectbox


def reset_conversation():
    # st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state["messages"] = []


st.sidebar.button('Clear', on_click=reset_conversation)

if "messages" not in st.session_state:
    # st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream = client.chat.completions.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = ""
        if add_selectbox != "claude-sonnet":
            st.write_stream(stream)
        else:
            for chunk in stream:
                try:
                    if chunk.delta:
                        # print(chunk.delta.text)
                        response = response + chunk.delta.text
                        message_placeholder.markdown(response + "| ")
                except Exception as e:
                    pass

        # remove the ending "|"
        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# streamlit run Chatbot.py
# if __name__ == "__main__":
#    main()