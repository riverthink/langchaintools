from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class PatientNoteSummary(BaseModel):
    summary: str = Field(..., description="Clinical synopsis of the patient note")
    problems: list[str] = Field(..., description="Key problems or concerns mentioned")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a clinician who writes concise, neutral patient note summaries."),
        ("user", "Summarize the patient note and list the key problems/concerns.\n\nNote:\n{note}"),
    ]
)

summarize_chain = prompt | ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(
    PatientNoteSummary
)

if __name__ == "__main__":
    sample_note = (
        "76-year-old female with hypertension and CKD stage 3. Presented with"
        " dizziness and BP 180/95. Started on amlodipine 5mg. Labs: creatinine"
        " 1.6 (baseline 1.5), potassium 4.8. No chest pain or neuro deficits."
    )

    result = summarize_chain.invoke({"note": sample_note})
    print(result)