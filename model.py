# from langchain_huggingface import (
#     ChatHuggingFace,
#     HuggingFacePipeline
# )

# # Step 1: Load model locally
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     device=-1,
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "do_sample": True,
#         "temperature": 0.5,
#         "return_full_text": False
#     }
# )

# # Step 2: Create Chat Model

# model = ChatHuggingFace(llm=llm)

# # Step 3: Ask question
# result = model.invoke(
#     "how many states are there in India?"
# )

# # Step 4: Print answer
# print(result.content)

from langchain_huggingface import HuggingFacePipeline

model = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 30,
        "return_full_text": False
    }
)

result = model.invoke("What is the capital of India? Answer:")

print(result)
