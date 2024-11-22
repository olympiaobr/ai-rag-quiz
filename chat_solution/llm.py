import os
from mistralai import Mistral
import logging

class LargeLanguageModel(object):
    def __init__(self, model="mistral-small-latest"):
        self.model = model
        self._api_key = os.environ.get("MISTRAL_API_KEY", None)
        if self._api_key is None:
            raise Exception(
                f"`MISTRAL_API_KEY` is None. Please set it in your environment variables."
            )
        self.client = Mistral(api_key=self._api_key)

    def call(self, prompt):
        # Catch rate limit error and retry 2 times with exponential backoff
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
                response_text = chat_response.choices[0].message.content
                return response_text
            except Exception as e:
                
                logging.error(f"Error happened while calling the model: {e}")
                if "Status 429" in str(e) or "Rate limit exceeded" in str(e):
                    logging.error(f"Rate limit error: {e}")
                    import time
                    time_to_wait = 2 ** attempt
                    logging.error(f"Waiting {time_to_wait} seconds before retrying")
                    time.sleep(time_to_wait)
                else:
                    logging.error(f"Api key: {self._api_key}")
                    raise e
        raise Exception("Rate limit exceeded after retries")