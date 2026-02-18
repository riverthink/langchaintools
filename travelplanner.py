from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class TravelPlan(BaseModel):
    overview: str = Field(..., description="One to two sentence trip summary")
    daily_plan: list[str] = Field(..., description="Day-by-day plan with key stops")
    reservations: list[str] = Field(..., description="Bookings to make ahead of time")
    packing: list[str] = Field(..., description="Weather- or activity-specific items")


brainstorm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a lively travel designer who brainstorms concise trip ideas."),
        (
            "user",
            "Destination: {destination}\n"
            "Days: {days}\n"
            "Travel style: {style}\n"
            "Constraints: {constraints}\n"
            "List three trip themes with a one-line rationale each.",
        ),
    ]
)

outline_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Turn the brainstorm into a realistic day-by-day outline."),
        (
            "user",
            "Traveler profile: {style}\n"
            "Draft ideas:\n{draft_plan}\n"
            "Create a numbered outline for each day with 2-3 anchor stops.",
        ),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Convert the outline into a concise, structured plan."),
        (
            "user",
            "Outline:\n{outline}\n"
            "Return fields: overview (1-2 sentences), daily_plan (list), "
            "reservations (list of must-book items), packing (list).",
        ),
    ]
)

long_travel_plan_chain = (  # Full end-to-end pipeline in one expression
    # 1) Start with user input and branch to generate a brainstorm draft.
    RunnablePassthrough.assign(
        draft_plan=brainstorm_prompt  # Build the brainstorm prompt with user fields
        | ChatOpenAI(model="gpt-4.1-mini", temperature=0.8)  # Creative brainstorm
        | StrOutputParser()  # Normalize brainstorm text
    )
    # 2) Use the brainstorm plus original inputs to build a day-by-day outline.
    | outline_prompt  # Feed both user inputs and brainstorm into outlining prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)  # More deterministic outline
    | StrOutputParser()  # Clean outline text for next stage
    # 3) Feed the outline into the final prompt and return a structured plan.
    | (lambda outline: {"outline": outline})  # Map outline into expected key
    | final_prompt  # Fill the final prompt template
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(TravelPlan)  # Typed output
)


if __name__ == "__main__":
    request = {
        "destination": "Lisbon, Portugal",
        "days": 3,
        "style": "food-loving traveler on a moderate budget who prefers to walk",
        "constraints": "Avoid long drives; keep evenings relaxed",
    }

    plan = long_travel_plan_chain.invoke(request)
    print(plan.model_dump_json(indent=2))
