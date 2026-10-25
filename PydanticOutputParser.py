from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json

load_dotenv()


class Student(BaseModel):
    name: str = Field(description="Student's full name")
    age: int = Field(gt=0, le=100, description="Student's age")
    education: str = Field(description="Current education")
    skills: list[str] = Field(description="Technical skills")
    experience_years: float = Field(
        ge=0,
        description="Years of professional experience"
    )
    career_goal: str = Field(description="Career goal")
    strengths: list[str] = Field(description="Personal strengths")


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


parser = PydanticOutputParser(pydantic_object=Student)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an assistant that extracts structured student information.

        Follow these instructions:
        - Extract information from the user's input.
        - Do not invent unknown personal details.
        - If a required field is missing, ask for clarification
          instead of inventing its value.
        - Return output in the required format.

        {format_instructions}
        """
    ),
    ("human", "{user_input}")
])


prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)


chain = prompt | model | parser


user_input = """
My name is Ali Ahmed.
I am 22 years old and studying BS Computer Science.
My skills are Python, SQL, Machine Learning and Pandas.
I have 1 year of professional experience.
My career goal is to become an AI Engineer.
My strengths are problem-solving and teamwork.
"""


try:
    
    result = chain.invoke({
        "user_input": user_input
    })

    print("Validated Pydantic Object:")
    print(result)


    json_output = result.model_dump_json(indent=4)

    print("\nJSON Output:")
    print(json_output)


except Exception as e:
    print("Error:", str(e))
