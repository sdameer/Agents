from typing import TypedDict, Annotated, Sequence, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class AState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str


# Calculator tools
@tool
def add(x: float, y: float) -> float:
    """Return the sum of two numbers."""
    return x + y


@tool
def sub(x: float, y: float) -> float:
    """Return the difference between two numbers."""
    return x - y


@tool
def mul(x: float, y: float) -> float:
    """Return the product of two numbers."""
    return x * y


@tool
def div(x: float, y: float) -> float:
    """Divide the first number by the second."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y


tools = [add, sub, mul, div]

model = ChatGroq(model="openai/gpt-oss-120b")


# 1. Router output schema
class RouteDecision(BaseModel):
    route: Literal["math", "general"] = Field(description="Choose math for calculations, general otherwise.")


router_model = model.with_structured_output(RouteDecision)


# 2. Router node
def router(state: AState):
    decision = router_model.invoke([
        SystemMessage(
            content=(
                "Classify the user's latest request. "
                "Choose 'math' for arithmetic or calculations. "
                "Choose 'general' for all other requests."
            )
        ),
        *state["messages"],
    ])

    return {"route": decision.route}


# 3. Math node: ReAct loop for calculator operations
math_model = model.bind_tools(tools)


def math_agent(state: AState):
    response = math_model.invoke([
        SystemMessage(
            content=(
                "You are a math assistant. Use the provided tools "
                "for calculations. For sequential operations, "
                "use each result in the next operation."
            )
        ),
        *state["messages"],
    ])

    return {"messages": [response]}


def should_continue(state: AState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end"


# 4. General-purpose node
def general_agent(state: AState):
    response = model.invoke([
        SystemMessage(
            content="You are a helpful general-purpose assistant."
        ),
        *state["messages"],
    ])

    return {"messages": [response]}


# 5. Build the graph
graph = StateGraph(AState)

graph.add_node("router", router)
graph.add_node("math", math_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("general", general_agent)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "math": "math",
        "general": "general",
    },
)

# Math node retains the ReAct tool loop.
graph.add_conditional_edges(
    "math",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "math")
graph.add_edge("general", END)

app = graph.compile()


# Test
res = app.invoke({
    "messages": [
        ("user", "Add 1 + 2, subtract 4, multiply by -1, then divide by 2.")
    ]
})

for message in res["messages"]:
    print(type(message).__name__, ":", message.content)
