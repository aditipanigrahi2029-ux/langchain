
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
model = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    device=-1,
    pipeline_kwargs={
        "max_new_tokens": 50,
        "return_full_text": False
    }
)

########### static prompt######
# result = model.invoke("What is Python?")
# print(result)


####### dynamic prompt###

prompt = PromptTemplate.from_template(
    "What is {topic}?"
)

chain = prompt | model

result = chain.invoke({"topic": "Machine Learning"})
print(result)