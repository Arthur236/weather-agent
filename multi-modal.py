from base64 import b64encode

from dotenv import load_dotenv

from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(
    model="gemma4",
    temperature=0.7
)

message = {
  "role": 'user',
  "content": [
    {
      "type": "text",
      "text": "Describe the contents of this image"
    },
    {
      "type": "image",
      "base64": b64encode(open("assets/image-1.jpeg", "rb").read()).decode("utf-8"),
      "mime_type": "image/jpeg"
    }
  ]
}

for chunk in model.stream([message]):
    print(chunk.text, end="", flush=True)
