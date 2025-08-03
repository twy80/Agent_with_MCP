import streamlit as st
import json
import os
import subprocess
import time
import shutil
from typing import Dict, Any, Optional, List, Type

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

# --- 1. MCP Process Management (Stable) ---

class MCPProcess:
    def __init__(self, server_name: str, command: str, args: List[str], env: Dict[str, str] = None):
        self.server_name, self.command, self.args, self.env = server_name, command, args, env or {}
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0

    def start(self, timeout: float = 10.0) -> bool:
        try:
            command_path = shutil.which(self.command)
            if not command_path: st.error(f"❌ Command not found: {self.command}"); return False
            full_env = {**os.environ, **self.env}
            self.process = subprocess.Popen([command_path] + self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=0, env=full_env)
            time.sleep(1)
            if self.process.poll() is not None: st.error(f"❌ Process '{self.server_name}' exited. Stderr: {self.process.stderr.read()}"); return False
            return self._initialize(timeout)
        except Exception as e: st.error(f"❌ Failed to start '{self.server_name}': {e}"); return False

    def _initialize(self, timeout: float = 5.0) -> bool:
        init_request = {"jsonrpc": "2.0", "id": self._next_id(), "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "langchain-mcp-client", "version": "3.0.0"}}}
        response = self._send_request(init_request, timeout)
        if response and "result" in response:
            self._send_notification({"jsonrpc": "2.0", "method": "notifications/initialized"})
            return True
        st.error(f"❌ Initialization failed for '{self.server_name}'. Response: {response}"); return False

    def list_tools(self) -> List[Dict[str, Any]]:
        response = self._send_request({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"})
        if response and "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            for tool in tools: tool['server'] = self.server_name
            return tools
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        response = self._send_request({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}, timeout=20)
        if not response: return "❌ No response from server (timeout)."
        if "error" in response: return f"❌ Server error: {response['error'].get('message', 'Unknown error')}"
        if "result" in response and "content" in response["result"]:
            return "\n".join(part["text"] for part in response["result"]["content"] if "text" in part)
        return "✅ Tool executed successfully (no output)."

    def _next_id(self) -> int: self.request_id += 1; return self.request_id

    def _send_request(self, request: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        if not self.process or self.process.poll() is not None: return None
        try:
            self.process.stdin.write(json.dumps(request) + "\n"); self.process.stdin.flush()
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            if response.get("id") == request["id"]:
                                return response
                        except json.JSONDecodeError: continue
                time.sleep(0.01)
            return None
        except Exception as e: st.error(f"❌ Send request failed: {e}"); return None

    def _send_notification(self, notification: Dict[str, Any]):
        if self.process and self.process.poll() is not None:
            try: self.process.stdin.write(json.dumps(notification) + "\n"); self.process.stdin.flush()
            except Exception as e: st.error(f"❌ Send notification failed: {e}")

    def stop(self):
        if self.process: self.process.terminate(); self.process = None

class MCPProcessManager:
    def __init__(self):
        self.processes: Dict[str, MCPProcess] = {}
        self.tools: List[Dict[str, Any]] = []

    def start_server(self, name: str, cfg: Dict[str, Any]) -> bool:
        if name in self.processes: self.processes[name].stop()
        proc = MCPProcess(name, cfg["command"], cfg.get("args", []), cfg.get("env", {}))
        if proc.start(): self.processes[name] = proc; return True
        return False

    def discover_tools(self):
        self.tools = [tool for process in self.processes.values() for tool in process.list_tools()]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        server_name = next((t.get('server') for t in self.tools if t.get('name') == tool_name), None)
        if not server_name or server_name not in self.processes: return f"❌ Tool '{tool_name}' not found."
        return self.processes[server_name].call_tool(tool_name, arguments)

    def stop_all(self):
        for process in self.processes.values(): process.stop()
        self.processes.clear(); self.tools = []

# --- 2. LangChain Integration (Corrected) ---

@st.cache_resource
def get_mcp_manager(): return MCPProcessManager()

mcp_manager = get_mcp_manager()

def get_python_type_from_json_schema(json_type: str) -> Type:
    return {"string": str, "number": float, "integer": int, "boolean": bool}.get(json_type, str)

def get_langchain_tools() -> List[StructuredTool]:
    lc_tools = []
    for tool_def in mcp_manager.tools:
        tool_name = tool_def["name"]
        schema = tool_def.get("inputSchema", {})
        required_args = schema.get("required", [])
        
        fields = {}
        for prop, info in schema.get("properties", {}).items():
            prop_type = get_python_type_from_json_schema(info.get("type", "string"))
            description = info.get("description")
            # CORRECTED: Check if the argument is required
            if prop in required_args:
                fields[prop] = (prop_type, Field(..., description=description))
            else:
                fields[prop] = (Optional[prop_type], Field(default=None, description=description))
        
        args_schema: Type[BaseModel] = create_model(f"{tool_name}Args", **fields)

        def _create_tool_func(name):
            def _tool_func(**kwargs):
                # Filter out None values for optional args not provided by the model
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                return mcp_manager.call_tool(name, filtered_kwargs)
            return _tool_func

        lc_tools.append(StructuredTool.from_function(
            name=tool_name,
            description=tool_def["description"],
            func=_create_tool_func(tool_name),
            args_schema=args_schema
        ))
    return lc_tools

def get_chat_model(provider: str, model_name: str):
    if provider == "OpenAI": return ChatOpenAI(model=model_name, temperature=0.7, streaming=True)
    if provider == "Anthropic": return ChatAnthropic(model_name=model_name, temperature=0.7, streaming=True)
    if provider == "Google Gemini": return ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
    raise ValueError(f"Unsupported provider: {provider}")

def create_langchain_agent(chat_model, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have access to the following tools. Use them when necessary."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# --- 3. Streamlit UI ---

def main():
    st.title("🤖 LangChain MCP Chatbot (Final Version)")

    if "messages" not in st.session_state: st.session_state.messages = []
    if "mcp_config_text" not in st.session_state: st.session_state.mcp_config_text = ""

    with st.sidebar:
        st.header("⚙️ Configuration")
        model_provider = st.selectbox("Select Model Provider", ["OpenAI", "Anthropic", "Google Gemini"])
        model_name = st.selectbox("Model", {"OpenAI": ["gpt-4o", "gpt-4o-mini"], "Anthropic": ["claude-3-5-sonnet-20240620"], "Google Gemini": ["gemini-2.5-flash", "gemini-2.5-pro"]}[model_provider])

        st.divider()
        st.subheader("🔧 MCP Server Configuration")
        config_option = st.radio("Config input method:", ["Text input", "File upload"], horizontal=True)
        
        if config_option == "File upload":
            uploaded_file = st.file_uploader("Upload mcpServers.json", type=['json'])
            if uploaded_file: st.session_state.mcp_config_text = uploaded_file.read().decode('utf-8')
        else:
            st.session_state.mcp_config_text = st.text_area("Paste mcpServers.json content", value=st.session_state.mcp_config_text, height=150)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect & Get Tools"):
                if st.session_state.mcp_config_text and st.session_state.mcp_config_text.strip():
                    try:
                        cfg = json.loads(st.session_state.mcp_config_text).get("mcpServers", {})
                        if not cfg: st.error("Invalid JSON or missing 'mcpServers' key.")
                        else:
                            mcp_manager.stop_all()
                            for name, server_cfg in cfg.items():
                                with st.spinner(f"Connecting to {name}..."):
                                    if mcp_manager.start_server(name, server_cfg):
                                        st.success(f"✅ Connected to {name}")
                                    else:
                                        st.error(f"❌ Failed to connect to {name}")
                            with st.spinner("Discovering tools..."): mcp_manager.discover_tools()
                    except json.JSONDecodeError as e: st.error(f"Invalid JSON provided: {e}")
                    except Exception as e: st.error(f"Connection failed: {e}")
                else:
                    st.warning("Please provide MCP server configuration.")
        with col2:
            if st.button("🔌 Disconnect All"):
                mcp_manager.stop_all()

        if mcp_manager.processes:
            st.success(f"✅ {len(mcp_manager.processes)} server(s) connected.")
            if mcp_manager.tools:
                st.info(f"🛠️ {len(mcp_manager.tools)} tools discovered.")
                with st.expander("View Tools"):
                    st.json([tool['name'] for tool in mcp_manager.tools])
            else:
                st.warning("Connected, but no tools were found.")
        
        st.divider()
        if st.button("🗑️ Clear Chat History"): st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("human").write(prompt)

        with st.chat_message("assistant"):
            try:
                chat_model = get_chat_model(model_provider, model_name)
                lc_tools = get_langchain_tools()
                agent_executor = create_langchain_agent(chat_model, lc_tools)
                
                output_placeholder = st.empty()
                full_response, tool_log = "", ""
                
                for chunk in agent_executor.stream({"input": prompt, "chat_history": st.session_state.messages[:-1]}):
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            tool_log += f"\n\n🔧 **Calling Tool:** `{action.tool}` with input `{action.tool_input}`\n"
                            output_placeholder.markdown(tool_log)
                    elif "steps" in chunk:
                        for step in chunk["steps"]:
                            tool_log += f"✅ **Tool Result:**\n```\n{step.observation}\n```"
                            output_placeholder.markdown(tool_log)
                    elif "output" in chunk:
                        output_chunk = chunk["output"]
                        if isinstance(output_chunk, str):
                            full_response += output_chunk
                        elif isinstance(output_chunk, list):
                            for block in output_chunk:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    full_response += block.get("text", "")
                        
                        # output_placeholder.markdown(tool_log + "\n\n---\n" + full_response + "▌")
                        output_placeholder.markdown(tool_log + "\n\n\n" + full_response + "▌")
                
                # output_placeholder.markdown(tool_log + "\n\n---" + full_response)
                output_placeholder.markdown(tool_log + "\n\n" + full_response)
                st.session_state.messages.append(AIMessage(content=full_response))

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    if st.sidebar.button("Finish"):
        os._exit(0)
