# Travel Planning Chain Demo

**Files:** [travelplanner.py](./travelplanner.py) | [Chain Diagram](./travelplanner_diagram.mermaid)

## Overview
This demo implements a multi-step travel planning system that demonstrates how LangChain chains can become complex when orchestrating multiple LLM calls with different roles and temperatures. It intentionally showcases the limitations of pure chain-based approaches, serving as motivation for adopting **LangGraph** for more complex orchestration needs.

## What This Demo Covers
- Multi-step chain orchestration
- Using different temperature settings for different tasks
- Structured outputs with Pydantic models
- Data passing between chain stages
- When to graduate from LangChain to LangGraph

## The Problem This Demo Illustrates

As your workflow grows more complex, pure chains face challenges:
- **Linear flow**: Chains execute in a fixed sequence (no branching, no loops)
- **No conditional logic**: Can't dynamically choose paths based on intermediate results
- **Limited state management**: Data passing becomes verbose and error-prone
- **Debugging complexity**: Hard to inspect intermediate outputs
- **No human-in-the-loop**: Can't pause for approvals or input mid-chain

This demo shows these limitations by implementing a relatively simple travel planning workflow that already feels complex.

## Architecture

The chain implements a three-stage pipeline:

```
User Input
    ↓
Stage 1: Brainstorm (Creative, temp=0.8)
    ↓
Stage 2: Outline (Balanced, temp=0.4)
    ↓
Stage 3: Structured Plan (Deterministic, temp=0.0)
    ↓
TravelPlan Output
```

Each stage has a distinct role, temperature, and prompt template.

## Code Structure

### 1. Output Schema
```python
class TravelPlan(BaseModel):
    overview: str = Field(..., description="One to two sentence trip summary")
    daily_plan: list[str] = Field(..., description="Day-by-day plan with key stops")
    reservations: list[str] = Field(..., description="Bookings to make ahead of time")
    packing: list[str] = Field(..., description="Weather- or activity-specific items")
```

Defines the final structured output with four fields:
- **overview**: High-level trip summary
- **daily_plan**: List of daily itineraries
- **reservations**: What to book in advance
- **packing**: What to bring

This ensures consistent, parseable outputs regardless of input variability.

### 2. Stage 1 - Brainstorming Prompt
```python
brainstorm_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a lively travel designer who brainstorms concise trip ideas."),
    ("user",
        "Destination: {destination}\n"
        "Days: {days}\n"
        "Travel style: {style}\n"
        "Constraints: {constraints}\n"
        "List three trip themes with a one-line rationale each."
    ),
])
```

**Purpose**: Generate creative trip themes

**Temperature**: 0.8 (high creativity)

**Input**: User's travel requirements (destination, days, style, constraints)

**Output**: Three possible trip themes with rationales

This stage encourages divergent thinking and exploration of different trip approaches.

### 3. Stage 2 - Outlining Prompt
```python
outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "Turn the brainstorm into a realistic day-by-day outline."),
    ("user",
        "Traveler profile: {style}\n"
        "Draft ideas:\n{draft_plan}\n"
        "Create a numbered outline for each day with 2-3 anchor stops."
    ),
])
```

**Purpose**: Convert creative themes into actionable daily outlines

**Temperature**: 0.4 (balanced creativity and structure)

**Input**: Traveler style + brainstorm draft from Stage 1

**Output**: Day-by-day numbered outline with 2-3 stops per day

This stage adds structure and feasibility to the creative brainstorm.

### 4. Stage 3 - Final Structured Plan
```python
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert the outline into a concise, structured plan."),
    ("user",
        "Outline:\n{outline}\n"
        "Return fields: overview (1-2 sentences), daily_plan (list), "
        "reservations (list of must-book items), packing (list)."
    ),
])
```

**Purpose**: Formalize the outline into the final `TravelPlan` schema

**Temperature**: 0.0 (deterministic, structured)

**Input**: Outline from Stage 2

**Output**: Structured `TravelPlan` object

This stage ensures consistent formatting and completeness.

### 5. The Complete Chain
```python
long_travel_plan_chain = (
    # Stage 1: Brainstorm
    RunnablePassthrough.assign(
        draft_plan=brainstorm_prompt
        | ChatOpenAI(model="gpt-4.1-mini", temperature=0.8)
        | StrOutputParser()
    )
    # Stage 2: Outline
    | outline_prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
    | StrOutputParser()
    # Stage 3: Structured output
    | (lambda outline: {"outline": outline})
    | final_prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(TravelPlan)
)
```

**Let's break this down step-by-step:**

#### Stage 1: Brainstorm
```python
RunnablePassthrough.assign(
    draft_plan=brainstorm_prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0.8)
    | StrOutputParser()
)
```

- `RunnablePassthrough.assign()`: Takes input and adds a new key (`draft_plan`) while keeping original keys
- Brainstorm prompt receives user input (destination, days, style, constraints)
- High temperature (0.8) for creative ideas
- `StrOutputParser()` converts LLM response to plain text
- **Output**: Original input dict + `draft_plan` key with brainstorm text

#### Stage 2: Outline
```python
| outline_prompt
| ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
| StrOutputParser()
```

- `outline_prompt` now has access to both original input AND `draft_plan`
- Medium temperature (0.4) balances creativity with structure
- **Output**: Raw outline text (no longer a dict!)

**Problem**: Notice how we've lost the original input dict—we now only have outline text.

#### Stage 3: Structured Plan
```python
| (lambda outline: {"outline": outline})
| final_prompt
| ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(TravelPlan)
```

- Lambda function wraps the outline string back into a dict with key `outline`
- Final prompt receives this dict
- Temperature 0 for deterministic structured output
- `.with_structured_output(TravelPlan)` enforces the Pydantic schema
- **Output**: `TravelPlan` object

### 6. Execution
```python
request = {
    "destination": "Lisbon, Portugal",
    "days": 3,
    "style": "food-loving traveler on a moderate budget who prefers to walk",
    "constraints": "Avoid long drives; keep evenings relaxed",
}

plan = long_travel_plan_chain.invoke(request)
print(plan.model_dump_json(indent=2))
```

Single invocation triggers all three stages sequentially.

## Key Patterns Demonstrated

### 1. Multi-Stage Pipelines
Different LLM calls with different:
- System prompts (roles)
- Temperature settings (creativity levels)
- Output formats (raw text vs structured)

### 2. Data Transformation Between Stages
- `RunnablePassthrough.assign()`: Add keys while preserving input
- `StrOutputParser()`: Convert LLM output to string
- Lambda functions: Reshape data for next stage

### 3. Variable Temperature Strategy
- **High temp (0.8)**: Exploration and creativity (brainstorming)
- **Medium temp (0.4)**: Balanced reasoning (outlining)
- **Low temp (0.0)**: Deterministic, structured output (final formatting)

### 4. Structured Output at the End
Only the final stage uses `.with_structured_output()` because:
- Early stages benefit from free-form creativity
- Final stage needs consistent, parseable format
- Enforcing structure too early can limit creativity

## Why This Gets Complex

Even this simple 3-step workflow shows complexity challenges:

### 1. Data Passing is Verbose
```python
| (lambda outline: {"outline": outline})
```
Manual reshaping of data between stages is error-prone and hard to maintain.

### 2. No Conditional Logic
What if the brainstorm is inadequate? The chain can't decide to:
- Retry with different prompts
- Request human input
- Branch to alternative paths

### 3. No Intermediate Inspection
Can't easily examine the brainstorm or outline without:
- Breaking the chain
- Adding custom callbacks
- Losing the elegant pipe syntax

### 4. Linear Execution Only
Can't implement:
- Loops (refine until quality threshold met)
- Parallel exploration (try multiple themes simultaneously)
- Human-in-the-loop approvals between stages

### 5. State Management
All state must flow through the pipe:
- Can't maintain separate state stores
- Can't persist intermediate results
- Hard to implement retry logic

## When to Use LangGraph Instead

If your workflow needs any of these, consider LangGraph:

### 1. Conditional Branching
```python
# LangGraph style (conceptual)
if brainstorm_quality_score < threshold:
    return "retry_brainstorm"
else:
    return "create_outline"
```

### 2. Loops and Iteration
```python
# LangGraph style (conceptual)
while not plan_is_satisfactory():
    refine_plan()
```

### 3. Human-in-the-Loop
```python
# LangGraph style (conceptual)
outline = create_outline()
if await human_approval(outline):
    finalize_plan()
else:
    modify_outline()
```

### 4. Parallel Execution
```python
# LangGraph style (conceptual)
themes = await parallel(
    brainstorm_cultural_theme(),
    brainstorm_food_theme(),
    brainstorm_adventure_theme()
)
```

### 5. State Persistence
LangGraph uses explicit state graphs:
- State is clearly defined
- Easy to inspect at any point
- Can persist and resume

## Running the Demo

```bash
python travelplanner.py
```

The script will:
1. Run the complete 3-stage chain
2. Print the final `TravelPlan` JSON
3. Show you how quickly chains become complex

### Sample Output Structure
```json
{
  "overview": "A 3-day food-focused exploration of Lisbon with walking tours...",
  "daily_plan": [
    "Day 1: Morning - Belém Tower and pastéis de nata at Pastéis de Belém...",
    "Day 2: Morning - Alfama district walking tour...",
    "Day 3: Morning - Time Out Market for breakfast..."
  ],
  "reservations": [
    "Dinner reservation at a fado restaurant",
    "Walking food tour booking"
  ],
  "packing": [
    "Comfortable walking shoes",
    "Light jacket for evenings",
    "Reusable water bottle"
  ]
}
```

## Experiment: Add More Complexity

Try adding:
1. **Weather check**: Look up weather and adjust packing list
2. **Budget calculation**: Estimate costs and adjust if over budget
3. **Availability check**: Verify restaurants/tours are open
4. **Quality gate**: Reject plans that don't meet criteria

You'll quickly find the chain becomes unmanageable—this is when you need LangGraph.

## Key Takeaways

### What Chains Do Well
- Simple linear workflows (2-3 steps)
- Fixed sequence operations
- Direct input → output transformations
- Prototyping and experimentation

### What Chains Struggle With
- Complex conditional logic
- Loops and iteration
- Human-in-the-loop workflows
- Parallel execution paths
- State management across steps
- Error recovery and retry logic

### When to Graduate to LangGraph
- More than 3-4 sequential steps
- Need for conditional branching
- Quality gates or validation between steps
- Human approval workflows
- Complex state management
- Production systems requiring observability

## Next Steps

1. **Study this chain carefully**: Understand each stage's role and data transformations
2. **Identify complexity**: Notice where the chain syntax becomes awkward
3. **Learn LangGraph**: Explore how LangGraph addresses these limitations
4. **Refactor this demo**: Try reimplementing this as a LangGraph state graph

## Conclusion

This demo intentionally pushes LangChain chains to show their limitations. The travel planner is simple conceptually but already shows complexity challenges. Real-world applications often need:
- Conditional logic
- Loops
- Human oversight
- Error handling
- State persistence

These requirements make LangGraph the better choice for production orchestration.

Use this demo as a reference point: if your workflow is more complex than this travel planner, strongly consider LangGraph from the start.
