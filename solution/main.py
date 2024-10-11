import os
from mistralai import Mistral

model = "mistral-small-latest"

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
print(chat_response.choices[0].message.content)