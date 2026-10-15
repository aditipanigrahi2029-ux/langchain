
from langchain_huggingface import HuggingFacePipeline

model = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    device=-1,
    pipeline_kwargs={
        "max_new_tokens": 50,
        "return_full_text": False
    }
)
result = model.invoke("What is Python?")
print(result)