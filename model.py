# model.py
import os
import openai

# Make sure OPENAI_API_KEY is set in your environment
openai.api_key = os.getenv("OPENAI_API_KEY")

class QAChain:
    def __init__(self, model="gpt-4-1-mini", max_tokens=500):
        self.model = model
        self.max_tokens = max_tokens

    def build_prompt(self, question, docs):
        """
        Creates a prompt by concatenating docs + question.
        """
        context = "\n\n---\n\n".join([doc["document"] for doc in docs])
        prompt = (
            f"You are a helpful AI assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

    def answer(self, question, docs):
        prompt = self.build_prompt(question, docs)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
            n=1,
        )

        answer = response['choices'][0]['message']['content'].strip()
        return answer