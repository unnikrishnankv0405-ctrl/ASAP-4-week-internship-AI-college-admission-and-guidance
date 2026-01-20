from transformers import pipeline
import os

# Load model using your token
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

prompt = """
Give college guidance for a student who wants Engineering in Tamil Nadu.
Explain simply and motivate the student.
"""

result = generator(prompt, max_length=200)
print(result[0]["generated_text"])
