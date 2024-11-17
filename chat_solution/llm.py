import os
from mistralai import Mistral


class LargeLanguageModel(object):
    def __init__(self, model="mistral-small-latest"):
        self.model = model
        api_key = os.environ.get("MISTRAL_API_KEY", None)
        if api_key is None:
            raise Exception(
                f"`MISTRAL_API_KEY` is None. Please set it in your environment variables."
            )
        self.client = Mistral(api_key=api_key)

    def call(self, prompt):

        # catch rate limit error and retry 2 time with 2 seconds delay
        # the error is a sdk error with the message "API error occurred: Status 429"
        # add exponential backoff  
        for attempt in range(1, 3):
            try:
                chat_response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
                return chat_response.choices[0].message.content
            except Exception as e:
                print(f"Error happended while calling the model: {e}")
                if "Status 429" in str(e) or "Rate limit exceeded" in str(e):
                    print(f"Rate limit error: {e}")
                    import time
                    time_to_wait = 2 ** attempt
                    print(f"Waiting {time_to_wait} seconds before retrying")
                    time.sleep(time_to_wait)

                    continue

        raise Exception("Rate limit exceeded")


if __name__ == "__main__":
    model = LargeLanguageModel()
    print(model.call("Who is the president of Germany"))
