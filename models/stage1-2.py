import os
import yfinance as yf
from typing import Annotated, TypedDict, Sequence, List, Union, Literal, Tuple
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

#Langgraph config
from langchain_core.runnables.config import RunnableConfig
config = RunnableConfig(recursion_limit=100)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_type: str | None
    # Planner
    plan: List[str]
    past_steps: Annotated[Sequence[BaseMessage], add_messages]
    response: str
    # Track which agent called the tool
    calling_agent: str | None
    # Track completed plan steps
    completed_steps: int

class AgentClassifier(BaseModel):
    agent_type: Literal["foundation_agent", "technical_agent", "macroeconomic_agent", "general_agent"] = Field(
        ...,
        description="Classify if the step requires macroeconomics (macroeconomics_agent), foundation (foundation_agent), technical (technical_agent) analysis or carry outgeneral task (general_agent) "
    )

class Plan(BaseModel):
    steps: List[str] = Field(
        ...,
        description="List of steps to follow that should be sorted in order"
    )

#Toolbox
@tool
def add(a: int, b: int):
    """Add two numbers"""
    return a + b

@tool
def web_search(query: str):
    """Web Search"""
    from tavily import TavilyClient
    client = TavilyClient(api_key=tavily_api_key)
    response = client.search(query)
    print(response)
    return response

#Fundamental Analysis Tools
@tool
def get_financial_statements(ticker: str):
    """Get financial statements"""
    tkr = yf.Ticker(ticker)
    return tkr.get_income_stmt(pretty=True, freq="yearly")


@tool
def get_balance_sheet(ticker: str):
    """Get balance sheet data"""
    tkr = yf.Ticker(ticker)
    return tkr.get_balance_sheet(pretty=True, freq="yearly")

@tool
def get_cash_flow(ticker: str):
    """Get cash flow data"""
    tkr = yf.Ticker(ticker)
    return tkr.get_cashflow(pretty=True, freq="yearly")

#Technical Analysis Tools
@tool
def get_ticker_data_3mo_daily(ticker: str):
    """Get ticker data for the last 3 months in daily intervals"""
    tkr = yf.Ticker(ticker)
    return tkr.history(period="3mo", interval="1d")

@tool
def get_ticker_data_10yr_monthly(ticker: str):
    """Get ticker data for the last 10 years in monthly intervals"""
    tkr = yf.Ticker(ticker)
    return tkr.history(period="10y", interval="1mo")

@tool
def get_ticker_data_1yr_weekly(ticker: str):
    """Get ticker data for the last 1 year in weekly intervals"""
    tkr = yf.Ticker(ticker)
    return tkr.history(period="1y", interval="1wk")


tools = [add, web_search, get_financial_statements, get_balance_sheet, get_cash_flow, get_ticker_data_3mo_daily, get_ticker_data_10yr_monthly, get_ticker_data_1yr_weekly]


#Initialize models
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0).bind_tools(tools)
structured_model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
answer_model = ChatOpenAI(model="gpt-5", temperature=0)


#Agent
def classify_agent(state: AgentState):
    completed = state.get("completed_steps", 0)
    try:
        step = state["plan"][completed]
    except IndexError:
        # If we've completed all steps, shouldn't reach here, but handle gracefully
        step = "Complete the remaining analysis"
    
    classifier_model = structured_model.with_structured_output(AgentClassifier)
    response = classifier_model.invoke([
        {
            "role": "system",
            "content": """
            Classify if the step is requires for either:
            - 'foundation_agent' : for stock foundation analysis
            - 'technical_agent': for stock technical analysis
            - 'macroeconomic_agent': for macroeconomic analysis
            - 'general_agent': for general task
            """
        },
        {"role": "user", "content": step}
    ])
    return {"agent_type": response.agent_type}

def agent_router(state: AgentState):
    message_type = state["agent_type"]
    if message_type == "foundation_agent":
        return {"next": "foundation_agent"}
    elif message_type == "technical_agent":
        return {"next": "technical_agent"}
    elif message_type == "macroeconomic_agent":
        return {"next": "macroeconomic_agent"}
    elif message_type == "general_agent":
        return {"next": "general_agent"}
    else:
        return {"next": "foundation_agent"}

#Planner
def planner(state: AgentState):
    last_message = state["messages"][-1]
    planner_model = structured_model.with_structured_output(Plan)
    response = planner_model.invoke([
        {
            "role": "system",
            "content": """
            You are a seasoned stock analyst with 30 years of experience. Based on the user's query, come up with a step by step plan to analyze the stock.
            This plan should be sorted in order and involve individual task, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"plan": response.steps}


def task_manager(state: AgentState):
    completed = state.get("completed_steps", 0)
    if completed >= len(state["plan"]):
        return {"next": "master_agent"}
    else:
        return {"next": "agent_classifier"}

#AI Agents
def master_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"You are a seasoned financial advisor with 30 years of experience. Based on the user's query and information provided, come up with a detailed answer to the user's query.Information provided: {state["past_steps"]}")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def foundation_agent(state: AgentState) -> AgentState:
    # Check if we have tool results to process
    messages = state["past_steps"]
    if messages and len(messages) >= 2:
        # Check if the last message is a tool result
        last_message = messages[-1]
        if hasattr(last_message, 'type') and last_message.type == 'tool':
            # Process tool results with the agent's context
            system_prompt = SystemMessage(content="""
                                          You are a fundamental analyst. 
                                          Process the tool results and provide your analysis or make additional tool calls if needed.
                                          Based on your analysis, either complete the current task or request additional tools.
                                          """)
            # Include the full conversation context
            response = model.invoke([system_prompt] + list(messages))
            # Check if agent completed the task (no tool calls means task is done)
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                completed = state.get("completed_steps", 0)
                return {"past_steps": list(messages) + [response], "calling_agent": "foundation_agent", "completed_steps": completed + 1}
            else:
                return {"past_steps": list(messages) + [response], "calling_agent": "foundation_agent"}
    
    # Initial task assignment - only calculate step index if not processing tool results
    completed = state.get("completed_steps", 0)
    try:
        step = state["plan"][completed]
    except IndexError:
        # If we've completed all steps, shouldn't reach here, but handle gracefully
        step = "Complete the current analysis"
    
    system_prompt = SystemMessage(content="""
                                  You are a fundamental analyst. 
                                  Based on the task assigned, use the tools available to you to carry out the task.
                                  """)
    task_message = HumanMessage(content=f"Task: {step}")
    response = model.invoke([system_prompt, task_message])
    # Check if agent completed the task (no tool calls means task is done)
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return {"past_steps": list(messages) + [response], "calling_agent": "foundation_agent", "completed_steps": completed + 1}
    else:
        return {"past_steps": list(messages) + [response], "calling_agent": "foundation_agent"}


def technical_agent(state: AgentState) -> AgentState:
    # Check if we have tool results to process
    messages = state["past_steps"]
    if messages and len(messages) >= 2:
        # Check if the last message is a tool result
        last_message = messages[-1]
        if hasattr(last_message, 'type') and last_message.type == 'tool':
            # Process tool results with the agent's context
            system_prompt = SystemMessage(content="""
                                          You are a seasoned technical analyst. 
                                          Process the tool results and provide your analysis or make additional tool calls if needed.
                                          Based on your analysis, either complete the current task or request additional tools.
                                          """)
            response = model.invoke([system_prompt] + list(messages))
            # Check if agent completed the task (no tool calls means task is done)
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                completed = state.get("completed_steps", 0)
                return {"past_steps": list(messages) + [response], "calling_agent": "technical_agent", "completed_steps": completed + 1}
            else:
                return {"past_steps": list(messages) + [response], "calling_agent": "technical_agent"}
    
    # Initial task assignment - only calculate step index if not processing tool results
    completed = state.get("completed_steps", 0)
    try:
        step = state["plan"][completed]
    except IndexError:
        # If we've completed all steps, shouldn't reach here, but handle gracefully
        step = "Complete the current analysis"
    
    system_prompt = SystemMessage(content="""
                                  You are a seasoned technical analyst. 
                                  Based on the task assigned, use the tools available to you to carry out the task.
                                  """)
    task_message = HumanMessage(content=f"Task: {step}")
    response = model.invoke([system_prompt, task_message])
    # Check if agent completed the task (no tool calls means task is done)
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return {"past_steps": list(messages) + [response], "calling_agent": "technical_agent", "completed_steps": completed + 1}
    else:
        return {"past_steps": list(messages) + [response], "calling_agent": "technical_agent"}


def macroeconomic_agent(state: AgentState) -> AgentState:
    # Check if we have tool results to process
    messages = state["past_steps"]
    if messages and len(messages) >= 2:
        # Check if the last message is a tool result
        last_message = messages[-1]
        if hasattr(last_message, 'type') and last_message.type == 'tool':
            # Process tool results with the agent's context
            system_prompt = SystemMessage(content="""
                                          You are a macroeconomics analyst. 
                                          Process the tool results and provide your analysis or make additional tool calls if needed.
                                          Based on your analysis, either complete the current task or request additional tools.
                                          """)
            response = model.invoke([system_prompt] + list(messages))
            # Check if agent completed the task (no tool calls means task is done)
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                completed = state.get("completed_steps", 0)
                return {"past_steps": list(messages) + [response], "calling_agent": "macroeconomic_agent", "completed_steps": completed + 1}
            else:
                return {"past_steps": list(messages) + [response], "calling_agent": "macroeconomic_agent"}
    
    # Initial task assignment - only calculate step index if not processing tool results
    completed = state.get("completed_steps", 0)
    try:
        step = state["plan"][completed]
    except IndexError:
        # If we've completed all steps, shouldn't reach here, but handle gracefully
        step = "Complete the current analysis"
    
    system_prompt = SystemMessage(content="""
                                  You are a macroeconomics analyst. 
                                  Based on the task assigned, use the tools available to you to carry out the task.
                                  """)
    task_message = HumanMessage(content=f"Task: {step}")
    response = model.invoke([system_prompt, task_message])
    # Check if agent completed the task (no tool calls means task is done)
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return {"past_steps": list(messages) + [response], "calling_agent": "macroeconomic_agent", "completed_steps": completed + 1}
    else:
        return {"past_steps": list(messages) + [response], "calling_agent": "macroeconomic_agent"}


def general_agent(state: AgentState) -> AgentState:
    # Check if we have tool results to process
    messages = state["past_steps"]
    if messages and len(messages) >= 2:
        # Check if the last message is a tool result
        last_message = messages[-1]
        if hasattr(last_message, 'type') and last_message.type == 'tool':
            # Process tool results with the agent's context
            system_prompt = SystemMessage(content="""
                                          You are a general agent. 
                                          Process the tool results and provide your analysis or make additional tool calls if needed.
                                          Based on your analysis, either complete the current task or request additional tools.
                                          """)
            response = model.invoke([system_prompt] + list(messages))
            # Check if agent completed the task (no tool calls means task is done)
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                completed = state.get("completed_steps", 0)
                return {"past_steps": list(messages) + [response], "calling_agent": "general_agent", "completed_steps": completed + 1}
            else:
                return {"past_steps": list(messages) + [response], "calling_agent": "general_agent"}
    
    # Initial task assignment - only calculate step index if not processing tool results
    completed = state.get("completed_steps", 0)
    try:
        step = state["plan"][completed]
    except IndexError:
        # If we've completed all steps, shouldn't reach here, but handle gracefully
        step = "Complete the current analysis"
    
    system_prompt = SystemMessage(content="""
                                  You are a general agent. 
                                  Based on the task assigned, use the tools available to you to carry out the task.
                                  """)
    task_message = HumanMessage(content=f"Task: {step}")
    response = model.invoke([system_prompt, task_message])
    # Check if agent completed the task (no tool calls means task is done)
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return {"past_steps": list(messages) + [response], "calling_agent": "general_agent", "completed_steps": completed + 1}
    else:
        return {"past_steps": list(messages) + [response], "calling_agent": "general_agent"}


def technical_agent_tool_calls(state: AgentState):
    messages = state["past_steps"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "task_manager"
    else:
        return "tools"
    

def foundation_agent_tool_calls(state: AgentState):
    messages = state["past_steps"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "task_manager"
    else:
        return "tools"
    

def macroeconomic_agent_tool_calls(state: AgentState):
    messages = state["past_steps"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "task_manager"
    else:
        return "tools"
    

def general_agent_tool_calls(state: AgentState):
    messages = state["past_steps"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "task_manager"
    else:
        return "tools"

def tool_router(state: AgentState):
    """Route back to the agent that called the tool"""
    calling_agent = state.get("calling_agent")
    if calling_agent:
        return calling_agent
    else:
        # Fallback to task_manager if no calling agent is tracked
        return "task_manager"


#Graph State Initialization
graph = StateGraph(AgentState)

# Custom Tool Handling Function
def handle_tools(state: AgentState):
    """Handle tool execution and return results"""
    # Get the last message from past_steps which should be an AIMessage with tool calls
    messages = state["past_steps"]
    if not messages:
        return {"past_steps": []}
    
    last_message = messages[-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"past_steps": messages}
    
    # Execute tool calls
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    tool_results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
                except Exception as e:
                    tool_results.append(ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_call["id"]
                    ))
                break
    
    # Add tool results to past_steps
    updated_past_steps = list(messages) + tool_results
    return {"past_steps": updated_past_steps}

#Tool Node Initialization
graph.add_node("tools", handle_tools)

#Agent Node Initialization
graph.add_node("master_agent", master_agent)
graph.add_node("foundation_agent", foundation_agent)
graph.add_node("technical_agent", technical_agent)
graph.add_node("macroeconomic_agent", macroeconomic_agent)
graph.add_node("general_agent", general_agent)
#Classifier Node Initialization
graph.add_node("agent_classifier", classify_agent)

#Router Node Initialization
graph.add_node("agent_router", agent_router)

#Planner Node Initialization
graph.add_node("planner", planner)

#Task Manager Node Initialization
graph.add_node("task_manager", task_manager)


graph.add_edge(START, "planner")
graph.add_edge("planner", "task_manager")

graph.add_conditional_edges(
    "task_manager",
    lambda state: state["next"],
    {
        "master_agent": "master_agent",
        "agent_classifier": "agent_classifier"
    }
)

graph.add_edge("agent_classifier", "agent_router")

graph.add_conditional_edges(
    "agent_router",
    lambda state: state["next"],
    {
        "foundation_agent": "foundation_agent",
        "technical_agent": "technical_agent",
        "macroeconomic_agent": "macroeconomic_agent",
        "general_agent": "general_agent"
    }
)

graph.add_conditional_edges(
    "foundation_agent",
    foundation_agent_tool_calls,
    {
        "task_manager": "task_manager",
        "tools": "tools"
    }
)

graph.add_conditional_edges(
    "technical_agent",
    technical_agent_tool_calls,
    {
        "task_manager": "task_manager",
        "tools": "tools"
    }
)

graph.add_conditional_edges(
    "macroeconomic_agent",
    macroeconomic_agent_tool_calls,
    {
        "task_manager": "task_manager",
        "tools": "tools"
    }
)

graph.add_conditional_edges(
    "general_agent",
    general_agent_tool_calls,
    {
        "task_manager": "task_manager",
        "tools": "tools"
    }
)

graph.add_conditional_edges(
    "tools",
    tool_router,
    {
        "foundation_agent": "foundation_agent",
        "technical_agent": "technical_agent", 
        "macroeconomic_agent": "macroeconomic_agent",
        "general_agent": "general_agent",
        "task_manager": "task_manager"
    }
)
graph.add_edge("master_agent", END)

app = graph.compile()

#Check Graph - Save as PNG file
try:
    graph_image = app.get_graph().draw_mermaid_png()
    with open("langgraph_workflow.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved as 'langgraph_workflow.png'")
except Exception as e:
    print(f"Could not save graph image: {e}")
    # Fallback: print mermaid diagram as text
    try:
        mermaid_text = app.get_graph().draw_mermaid()
        print("\nMermaid Diagram:")
        print(mermaid_text)
    except Exception as e2:
        print(f"Could not generate mermaid diagram: {e2}")

# Proper stream debugging functions
def print_stream_debug(stream):
    """
    Enhanced stream debugging function that shows the workflow step by step
    """
    print("=== LANGGRAPH WORKFLOW DEBUG ===")
    print("Starting stream processing...\n")
    
    for i, chunk in enumerate(stream):
        print(f"--- STEP {i+1} ---")
        
        # Print the current node being executed
        if "agent_type" in chunk:
            print(f"Agent Type: {chunk['agent_type']}")
        
        # Print the plan if available
        if "plan" in chunk:
            print(f"Plan: {chunk['plan']}")
        
        # Print messages
        if "messages" in chunk:
            print("Messages:")
            for msg in chunk["messages"]:
                if hasattr(msg, 'content'):
                    print(f"  - {type(msg).__name__}: {msg.content[:100]}...")
                else:
                    print(f"  - {type(msg).__name__}: {msg}")
        
        # Print past steps and completed counter
        if "past_steps" in chunk:
            completed = chunk.get('completed_steps', 0)
            print(f"Past Steps: {len(chunk['past_steps'])} messages, {completed} steps completed")
        
        # Print next action if available
        if "next" in chunk:
            print(f"Next Action: {chunk['next']}")
        
        print()  # Empty line for readability
    # Print the final result if available
    # (This line had a bug, just removing it)
    print("=== STREAM PROCESSING COMPLETE ===")


query = "buy or sell JPM stock"

print("\nSTARTING STREAM...")
print("=" * 60)
print_stream_debug(app.stream({"messages": [HumanMessage(content=query)], "completed_steps": 0}, stream_mode="values", config=config))
print("\n" + "=" * 60)
print("Goodbye!")
