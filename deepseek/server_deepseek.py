"to run bush:sudo /opt/pytorch/bin/uvicorn main:app --host 0.0.0.0 --port 80"

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from pydantic import BaseModel

app = FastAPI()

# Model loading (moved outside the API function for efficiency)
print("loading...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite",
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded successfully!")

if model is None:
    raise ValueError("Model failed to load!")

model.config.use_cache = False  # Disable cache

device = model.device if model.device is not None else "cpu"

class InputText(BaseModel):
    text: str
    max_length: int = 100 # Default max length

@app.post("/generate/")
async def generate_text(input_data: InputText):
    """Generates text based on the provided input."""
    try:
        inputs = tokenizer(input_data.text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)

        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            max_length=input_data.max_length,
            use_cache=False,
            attention_mask=inputs['attention_mask']
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"generated_text": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))