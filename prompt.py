
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
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

# prompt = PromptTemplate.from_template(
#     "What is {topic}?"
# )

# chain = prompt | model
# result = chain.invoke({"topic": "Machine Learning"})
# print(result)

######### prompt templeste########

# prompt = PromptTemplate(
#     input_variables=["topic","lanuage"],
#     template= "explain {topic} in this {lanuage}"
# )
# chain = prompt | model
# result = chain.invoke({
#     "topic": "AI",
#     "lanuage":"Hindi"
# })
# print(result)

######chatPromptTemplate###
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful teacher."),
#     ("human", "Explain {topic} in simple words.")
# ])

# chain = prompt | model

# result = chain.invoke({
#     "topic": "Deep Learning"
# })

# print(result)

######systemmsg, humanmsg, aimsg###
messages=[
    SystemMessage(content="u r a teacher"),
    HumanMessage(content="what is python"),
    #AIMessage(content="python is miodel"),
    HumanMessage(content="give one example")

]
result = model.invoke(messages)
print(result)
ai_msg= AIMessage(content=result)
print(ai_msg.content)
# for m in messages:
#     print(m.type,":",m.content)