import os
from crewai import Agent, Task, Crew
from litellm import completion  # LiteLLM wrapper for Gemini

class GeminiLLM:
    def __init__(self):
        self.model_name = "google/gemini-pro"
        self.api_key = os.getenv("GEMINI_API_KEY")  # Ensure API key is set in environment variables
        
        if not self.api_key:
            raise ValueError("❌ ERROR: GEMINI_API_KEY is missing! Set it in your environment.")

    def __call__(self, prompt: str) -> str:
        """ Function that makes this class callable, so it works like an LLM in CrewAI. """
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"
