import os
from google import genai
from PIL import Image

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Example inputs
image_path = "diagram.png"
table_text = """
Part No | Description    | Qty
--------------------------------
101     | Gear Assembly  | 2
102     | Shaft Bearing  | 4
103     | Housing Plate  | 1
"""

# Load image
image = Image.open(image_path)

# Prompt for augmentation
prompt = f"""
You are a data augmentation assistant.

Inputs:
1. Diagram (attached image)
2. Table Text:
\"\"\"
{table_text}
\"\"\"

Task:
- Generate 2 augmented training samples.
- Each sample must include:
  - A paraphrased version of the table text (same content, different wording/format).
  - 2 alternative captions for the diagram.
  - 1 question-answer pair that connects table info with the diagram.
- Output JSON list with keys: augmented_table, captions, qa.
"""

# Call Gemini
response = client.models.generate_content(
    model="gemini-2.0-flash",   # multimodal, fast
    contents=[prompt, image]
)

print(response.text)
