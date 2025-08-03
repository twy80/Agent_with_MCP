import streamlit as st
import json
import os
import subprocess
import time
import shutil
from typing import Dict, Any, Generator, Optional, List
import openai
import anthropic
import google.generativeai as genai


st.set_page_config(
    page_title="Multi-Model Chatbot",
    page_icon="🤖",
    layout="wide"
)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mcp_tools" not in st.session_state:
        st.session_state.mcp_tools = []
    if "mcp_processes" not in st.session_state:
        st.session_state.mcp_processes = {}
    if "connected_servers" not in st.session_state:
        st.session_state.connected_servers = []
    if 'mcp_manager' not in st.session_state:
        st.session_state.mcp_manager = MCPProcessManager()

def validate_json(json_str: str) -> tuple[bool, Dict[str, Any]]:
    try:
        data = json.loads(json_str)
        return True, data
    except json.JSONDecodeError as e:
        return False, {"error": str(e)}

# Direct subprocess-based MCP communication
class MCPProcess:
    def __init__(self, server_name: str, command: str, args: List[str], env: Dict[str, str] = None):
        self.server_name = server_name
        self.command = command
        self.args = args
        self.env = env or {}
        self.process = None
        self.request_id = 0

    def start(self, timeout: float = 10.0) -> bool:
        """Start the MCP server process with timeout"""
        try:
            # Check if command exists
            if not shutil.which(self.command):
                st.warning(f"⚠️ Command '{self.command}' not found for '{self.server_name}'")
                return False

            # Start the process
            full_env = {**os.environ, **self.env}
            start_time = time.time()

            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered
                env=full_env
            )

            # Wait a moment for process to start
            time.sleep(0.5)

            # Check if process started successfully
            if self.process.poll() is not None:
                st.error(f"❌ Process '{self.server_name}' exited immediately")
                return False

            # Initialize MCP connection with timeout
            if self.initialize(timeout - (time.time() - start_time)):
                return True
            else:
                st.error(f"❌ Connection timeout for '{self.server_name}' after {timeout}s")
                self.stop()
                return False

        except Exception as e:
            st.error(f"❌ Failed to start '{self.server_name}': {str(e)}")
            return False

    def initialize(self, timeout: float = 5.0) -> bool:
        """Initialize MCP connection with timeout"""
        try:
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "streamlit-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }

            response = self._send_request(init_request, timeout)
            if response and "result" in response:
                # Send initialized notification
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                self._send_notification(initialized_notification)
                return True
            return False

        except Exception as e:
            st.error(f"❌ Initialize failed for '{self.server_name}': {str(e)}")
            return False

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list"
            }

            response = self._send_request(request)
            if response and "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                # Add server name to each tool
                for tool in tools:
                    tool['server'] = self.server_name
                return tools
            return []

        except Exception as e:
            st.error(f"❌ List tools failed for '{self.server_name}': {str(e)}")
            return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            # Debug output (only if debug mode is enabled)
            if st.session_state.get('debug_mode', False):
                st.write(f"🔍 Debug: Calling tool '{tool_name}' with args: {arguments}")
                st.write(f"🔍 Debug: Full request: {json.dumps(request, indent=2)}")

            response = self._send_request(request)

            # Debug: Show response
            if response and st.session_state.get('debug_mode', False):
                st.write(f"🔍 Debug: Response: {json.dumps(response, indent=2)}")
            if not response:
                return f"❌ No response from server (timeout after 10s)"

            if "error" in response:
                error = response["error"]
                return f"❌ Server error: {error.get('message', 'Unknown error')}"

            if response and "result" in response:
                result = response["result"]
                if "content" in result:
                    content_parts = []
                    for content in result["content"]:
                        if isinstance(content, dict) and "text" in content:
                            content_parts.append(content["text"])
                        else:
                            content_parts.append(str(content))
                    return "\n".join(content_parts)
                else:
                    return "✅ Tool executed successfully (no output)"
            elif response and "error" in response:
                return f"❌ Tool error: {response['error'].get('message', 'Unknown error')}"
            else:
                return "❌ No response from tool"

        except Exception as e:
            return f"❌ Tool execution failed: {str(e)}"

    def _next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    def _send_request(self, request: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Send request and wait for response"""
        if not self.process or self.process.poll() is not None:
            return None

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()

            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            if "id" in response and response["id"] == request["id"]:
                                return response
                        except json.JSONDecodeError:
                            continue
                time.sleep(0.01)  # Small delay to prevent busy waiting

            return None

        except Exception as e:
            st.error(f"❌ Send request failed: {str(e)}")
            return None

    def _send_notification(self, notification: Dict[str, Any]):
        """Send notification (no response expected)"""
        if not self.process or self.process.poll() is not None:
            return

        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line)
            self.process.stdin.flush()
        except Exception as e:
            st.error(f"❌ Send notification failed: {str(e)}")

    def stop(self):
        """Stop the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
            self.process = None

# Global process manager
class MCPProcessManager:
    def __init__(self):
        self.processes = {}

    def start_server(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """Start a single MCP server"""
        if "command" not in server_config:
            st.error(f"❌ Server '{server_name}' missing command")
            return False

        command = server_config["command"]
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        # Stop existing process if any
        if server_name in self.processes:
            self.processes[server_name].stop()

        # Start new process with timeout
        process = MCPProcess(server_name, command, args, env)
        if process.start(timeout=15.0):  # 15 second timeout per server
            self.processes[server_name] = process
            return True
        return False

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get tools from all connected servers"""
        all_tools = []
        for process in self.processes.values():
            tools = process.list_tools()
            all_tools.extend(tools)
        return all_tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], server_name: str = None) -> str:
        """Call a tool from the appropriate server"""
        if server_name and server_name in self.processes:
            return self.processes[server_name].call_tool(tool_name, arguments)

        # Find server with this tool
        for process in self.processes.values():
            tools = process.list_tools()
            if any(tool.get('name') == tool_name for tool in tools):
                return process.call_tool(tool_name, arguments)

        return f"❌ Tool '{tool_name}' not found in any connected server"

    def stop_all(self):
        """Stop all MCP servers"""
        for process in self.processes.values():
            process.stop()
        self.processes.clear()

def connect_all_mcp_servers(servers_config: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Connect to all MCP servers"""
    manager = st.session_state.mcp_manager
    connected_servers = {}

    for server_name, server_config in servers_config.items():
        if manager.start_server(server_name, server_config):
            connected_servers[server_name] = server_config

    # Get all tools
    all_tools = manager.get_all_tools()

    # Show results
    if not connected_servers:
        st.error("❌ Failed to connect to any MCP servers")

    return connected_servers, all_tools

def stream_openai_response_with_tools(client, messages: list, model: str, mcp_tools: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Stream OpenAI response with MCP tool support"""
    try:
        openai_tools = convert_mcp_tools_to_openai_format(mcp_tools) if mcp_tools else None

        # Debug: Show converted OpenAI tools
        if st.session_state.get('debug_mode', False) and openai_tools:
            yield f"🔍 **Debug:** Converted {len(openai_tools)} tools for OpenAI\n"
            yield f"🔍 **Debug:** First tool: {openai_tools[0]['function']['name']}\n\n"

        # First call to potentially get tool calls
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto" if openai_tools else None,
            temperature=0.7
        )

        message = response.choices[0].message

        # Check for tool calls
        if message.tool_calls:
            # Process tool calls
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                yield f"\n\n🔧 **Executing MCP tool: {tool_name}**\n\n"

                if st.session_state.get('debug_mode', False):
                    yield f"🔍 **Debug:** Tool call detected - {tool_name} with args: {tool_args}\n\n"

                # Execute the tool
                manager = st.session_state.mcp_manager
                result = manager.call_tool(tool_name, tool_args)

                yield f"✅ **Result:**\n"

                # Truncate very long results for better UI
                if len(result) > 2000:
                    truncated_result = result[:2000] + "\n... (결과가 너무 길어 일부만 표시됩니다)"
                    yield f"```\n{truncated_result}\n```\n\n"
                else:
                    yield f"```\n{result}\n```\n\n"

                # Add tool call and result to message history
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    }]
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Generate follow-up response with tool results
            yield f"📝 **Analyzing results...**\n\n"

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            # No tool calls, make a streaming call
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto" if openai_tools else None,
                stream=True,
                temperature=0.7
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"Error: {str(e)}"

def stream_openai_response(client, messages: list, model: str) -> Generator[str, None, None]:
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error: {str(e)}"

def convert_mcp_tools_to_openai_format(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tools to OpenAI function calling format"""
    openai_tools = []
    for tool in mcp_tools:
        input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

        # Ensure schema has required structure
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}

        # Ensure properties exists
        if "properties" not in input_schema:
            input_schema["properties"] = {}

        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", "No description"),
                "parameters": input_schema
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools

def convert_mcp_tools_to_anthropic_format(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tools to Anthropic tool format"""
    anthropic_tools = []
    for tool in mcp_tools:
        input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

        # Ensure schema has required structure
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}

        # Ensure properties exists
        if "properties" not in input_schema:
            input_schema["properties"] = {}

        anthropic_tool = {
            "name": tool.get("name", "unknown"),
            "description": tool.get("description", "No description"),
            "input_schema": input_schema
        }
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools

def stream_anthropic_response_with_tools(client, messages: list, model: str, mcp_tools: List[Dict[str, Any]]) -> Generator[str, None, None]:
    try:
        system_message = ""
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        anthropic_tools = convert_mcp_tools_to_anthropic_format(mcp_tools) if mcp_tools else None

        # Debug: Show converted Anthropic tools
        if st.session_state.get('debug_mode', False) and anthropic_tools:
            yield f"🔍 **Debug:** Converted {len(anthropic_tools)} tools for Anthropic\n"
            yield f"🔍 **Debug:** First tool: {anthropic_tools[0]['name']}\n\n"

        # Use streaming for better UX
        with client.messages.stream(
            model=model,
            max_tokens=4000,
            system=system_message,
            messages=chat_messages,
            tools=anthropic_tools,
            temperature=0.7
        ) as stream:
            message_content = []
            for text in stream.text_stream:
                yield text
                message_content.append(text)
            
            # Get the final message for tool processing
            message = stream.get_final_message()

        # Check for tool use in the response
        tool_calls_found = False
        tool_results = []

        for content_block in message.content:
            if content_block.type == 'text':
                yield content_block.text
            elif content_block.type == 'tool_use':
                tool_calls_found = True
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id

                yield f"\n\n🔧 **Executing MCP tool: {tool_name}**\n\n"

                # Debug: Show tool call details
                if st.session_state.get('debug_mode', False):
                    yield f"🔍 **Debug:** Tool call detected - {tool_name} with args: {tool_input}\n\n"

                # Find which server has this tool
                server_name = None
                for tool in mcp_tools:
                    if tool.get('name') == tool_name:
                        server_name = tool.get('server')
                        break

                manager = st.session_state.mcp_manager
                result = manager.call_tool(tool_name, tool_input, server_name)

                yield f"✅ **Result:**\n"

                # Truncate very long results for better UI
                if len(result) > 2000:
                    truncated_result = result[:2000] + "\n... (결과가 너무 길어 일부만 표시됩니다)"
                    yield f"```\n{truncated_result}\n```\n\n"
                else:
                    yield f"```\n{result}\n```\n\n"

                # Store tool result for follow-up
                tool_results.append({
                    "tool_use_id": tool_use_id,
                    "content": result
                })

        # If tools were called, generate a follow-up response with the results
        if tool_calls_found and tool_results:
            yield f"📝 **Analyzing results...**\n\n"

            # Create a new conversation with tool results
            follow_up_messages = chat_messages.copy()
            follow_up_messages.append({
                "role": "assistant", 
                "content": message.content
            })
            follow_up_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result["tool_use_id"],
                        "content": tool_result["content"]
                    } for tool_result in tool_results
                ]
            })

            # Generate follow-up response
            follow_up_message = client.messages.create(
                model=model,
                max_tokens=4000,
                system=system_message + "\n\nBased on the tool results above, provide a comprehensive answer to the user's question.",
                messages=follow_up_messages,
                temperature=0.7
            )

            for content_block in follow_up_message.content:
                if content_block.type == 'text':
                    yield content_block.text

        # Debug: Show if no tool calls were made
        elif not tool_calls_found and mcp_tools and st.session_state.get('debug_mode', False):
            yield "\n\n⚠️ **Debug:** No tool calls detected by Anthropic despite having available tools.\n"

    except Exception as e:
        yield f"Error: {str(e)}"

def stream_anthropic_response(client, messages: list, model: str) -> Generator[str, None, None]:
    try:
        system_message = ""
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        with client.messages.stream(
            model=model,
            max_tokens=4000,
            system=system_message,
            messages=chat_messages,
            temperature=0.7
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Error: {str(e)}"

def clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Clean schema to remove unsupported fields for Gemini"""
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        # Remove unsupported fields
        if key in ['minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf', 'format', 'additionalProperties', '$schema', '$id', '$ref', '$defs', 'definitions', 'default', 'title', 'minLength']:
            continue

        if isinstance(value, dict):
            cleaned[key] = clean_schema_for_gemini(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_for_gemini(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value

    return cleaned

def convert_mcp_tools_to_gemini_format(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tools to Gemini function calling format"""
    gemini_tools = []
    for tool in mcp_tools:
        input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

        # Ensure schema has required structure
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}

        # Clean schema for Gemini compatibility
        cleaned_schema = clean_schema_for_gemini(input_schema)

        # Ensure properties exists
        if "properties" not in cleaned_schema:
            cleaned_schema["properties"] = {}

        gemini_tool = {
            "function_declarations": [{
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", "No description"),
                "parameters": cleaned_schema
            }]
        }
        gemini_tools.append(gemini_tool)
    return gemini_tools

def stream_gemini_response_with_tools(model, messages: list, mcp_tools: List[Dict[str, Any]]) -> Generator[str, None, None]:
    try:
        gemini_tools = convert_mcp_tools_to_gemini_format(mcp_tools) if mcp_tools else None

        # Debug: Show converted Gemini tools
        if st.session_state.get('debug_mode', False) and gemini_tools:
            yield f"🔍 **Debug:** Converted {len(gemini_tools)} tools for Gemini\n"
            yield f"🔍 **Debug:** First tool: {gemini_tools[0]['function_declarations'][0]['name']}\n\n"

        # Prepare chat history
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})

        # Start chat 
        chat = model.start_chat(history=chat_history)

        # Send message with tools if available, always use streaming
        if gemini_tools:
            response = chat.send_message(messages[-1]["content"], tools=gemini_tools, stream=True)
        else:
            response = chat.send_message(messages[-1]["content"], stream=True)

        # Process streaming response
        function_calls = []
        text_parts = []
        
        for chunk in response:
            if chunk and hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part and hasattr(part, 'function_call') and part.function_call:
                            # Function call detected
                            function_calls.append(part.function_call)
                        elif part and hasattr(part, 'text') and part.text:
                            # Regular text response
                            yield part.text
                            text_parts.append(part.text)
        
        # Process function calls if any
        if function_calls:
            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args)

                yield f"\n\n🔧 **Executing MCP tool: {tool_name}**\n\n"

                if st.session_state.get('debug_mode', False):
                    yield f"🔍 **Debug:** Tool call detected - {tool_name} with args: {tool_args}\n\n"

                # Execute the tool
                manager = st.session_state.mcp_manager
                result = manager.call_tool(tool_name, tool_args)

                yield f"✅ **Result:**\n"

                # Truncate very long results for better UI
                if len(result) > 2000:
                    truncated_result = result[:2000] + "\n... (결과가 너무 길어 일부만 표시됩니다)"
                    yield f"```\n{truncated_result}\n```\n\n"
                else:
                    yield f"```\n{result}\n```\n\n"

                # Send function response back to get final answer
                yield f"📝 **Analyzing results...**\n\n"

                function_response = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result}
                    )
                )

                final_response = chat.send_message([function_response], stream=True)
                for chunk in final_response:
                    if chunk.text:
                        yield chunk.text

    except Exception as e:
        yield f"Error: {str(e)}"

def stream_gemini_response(model, messages: list) -> Generator[str, None, None]:
    try:
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(messages[-1]["content"], stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {str(e)}"

def main():
    st.title("🤖 Multi-Model Chatbot")

    init_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model provider selection - OpenAI now supported with MCP
        model_provider = st.selectbox(
            "Select Model Provider",
            ["OpenAI", "Anthropic", "Google Gemini"],
            key="model_provider"
        )

        if model_provider == "OpenAI":
            model_name = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                key="openai_model"
            )
        elif model_provider == "Anthropic":
            model_name = st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                key="anthropic_model"
            )
        else:  # Google Gemini
            model_name = st.selectbox(
                "Model",
                ["gemini-2.5-flash", "gemini-2.5-pro"],
                key="gemini_model"
            )

        st.divider()

        st.subheader("🔧 MCP Server Configuration")

        # Add file upload option
        config_option = st.radio(
            "Configuration input method:",
            ["Text input", "File upload"],
            horizontal=True
        )

        if config_option == "File upload":
            uploaded_file = st.file_uploader(
                "Upload mcpServers.json file:",
                type=['json'],
                accept_multiple_files=False
            )

            if uploaded_file is not None:
                try:
                    mcp_servers_input = uploaded_file.read().decode('utf-8')
                    st.text_area(
                        "Loaded configuration:",
                        value=mcp_servers_input,
                        height=200,
                        disabled=True,
                        key="loaded_config_display"
                    )
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    mcp_servers_input = ""
            else:
                mcp_servers_input = ""
        else:
            mcp_servers_input = st.text_area(
                "Paste mcpServers.json content:",
                height=200,
                placeholder='{"mcpServers": {"server-name": {"command": "...", "args": [...]}}}',
                key="mcp_servers_input"
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔌 Connect All MCP Servers"):
                if mcp_servers_input.strip():
                    is_valid, config = validate_json(mcp_servers_input)
                    if is_valid and "mcpServers" in config:
                        try:
                            servers, tools = connect_all_mcp_servers(config["mcpServers"])
                            st.session_state.mcp_processes = servers
                            st.session_state.mcp_tools = tools
                            st.session_state.connected_servers = list(servers.keys())
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
                    else:
                        st.error("❌ Invalid JSON or missing 'mcpServers' key")
                else:
                    st.warning("⚠️ Please enter valid server configuration")

        with col2:
            if st.button("🔌 Disconnect All MCP"):
                st.session_state.mcp_manager.stop_all()
                st.session_state.mcp_processes = {}
                st.session_state.mcp_tools = []
                st.session_state.connected_servers = []
                st.info("🔌 All MCP servers disconnected")

        # Add connection test button
        if st.session_state.mcp_tools:
            if st.button("🧪 Test MCP Connection"):
                st.write("Testing MCP server connections...")
                manager = st.session_state.mcp_manager

                for server_name, process in manager.processes.items():
                    if process.process and process.process.poll() is None:
                        # Test with a simple tool list request
                        tools = process.list_tools()
                        if tools:
                            st.success(f"✅ {server_name}: Connected ({len(tools)} tools available)")
                        else:
                            st.error(f"❌ {server_name}: Connected but no tools found")
                    else:
                        st.error(f"❌ {server_name}: Process not running")

        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False))

        if st.session_state.mcp_tools:
            st.success(f"🛠️ Available MCP tools: {len(st.session_state.mcp_tools)} from {len(st.session_state.connected_servers)} servers")
            with st.expander("🔍 View MCP Tools"):
                # Group tools by server
                by_server = {}
                for tool in st.session_state.mcp_tools:
                    server = tool.get('server', 'unknown')
                    if server not in by_server:
                        by_server[server] = []
                    by_server[server].append(tool)

                # Display each server with tools
                for i, (server, tools) in enumerate(by_server.items()):
                    st.markdown(f"🟢 **{server}** - Ready ({len(tools)} tool{'s' if len(tools) != 1 else ''})")

                    # Sort tools alphabetically and display in compact format
                    tool_names = sorted([tool.get('name', 'Unknown') for tool in tools])
                    tools_text = "  " + ", ".join(tool_names)
                    st.markdown(tools_text)

                    # Add spacing between servers (except for the last one)
                    if i < len(by_server) - 1:
                        st.markdown("")

        st.divider()

        st.subheader("📝 Additional Context")
        additional_context = st.text_area(
            "Enter additional context or data:",
            height=100,
            placeholder='Any additional information to provide to the AI...',
            key="additional_context"
        )

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare messages for API
        messages_for_api = st.session_state.messages.copy()

        # Add system messages
        system_messages = []

        if st.session_state.mcp_tools:
            tools_info = (
                "Your goal is to provide answers to the user's inquiries.\n\n"
                "You have access to the following MCP (Model Context Protocol) tool(s):\n"
            )
            for tool in st.session_state.mcp_tools:
                name = tool.get('name', 'Unknown')
                desc = tool.get('description', 'No description')
                schema = tool.get('inputSchema', {})

                # Show required parameters from the actual MCP schema
                required_params = schema.get('required', []) if isinstance(schema, dict) else []
                if required_params:
                    tools_info += f"- {name}: {desc} (Required: {', '.join(required_params)})\n"
                else:
                    tools_info += f"- {name}: {desc}\n"

            tools_info += (
                "\nCall tool(s) only when the user's request requires accessing external "
                "data or performing actions that you cannot do with your built-in knowledge. "
                "When giving your answers, tell the user what your response "
                "is based on and which tools you use. Use Markdown syntax "
                "and include relevant sources, such as links (URLs), following "
                "MLA format. If the information is not available, inform "
                "the user explicitly that the answer could not be found. "
            )
            system_messages.append(tools_info)

        additional_context = st.session_state.get("additional_context", "")
        if additional_context.strip():
            system_messages.append(f"Additional Context: {additional_context}")

        if system_messages:
            system_message = "\n\n".join(system_messages)
            messages_for_api.insert(0, {"role": "system", "content": system_message})

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Debug: Show MCP tools being passed to OpenAI
            if st.session_state.get('debug_mode', False) and st.session_state.mcp_tools:
                st.write(f"🔍 **Debug:** {len(st.session_state.mcp_tools)} MCP tools available")
                st.write("Tools being passed to OpenAI:")
                for i, tool in enumerate(st.session_state.mcp_tools[:3]):  # Show first 3 tools
                    st.write(f"- {tool.get('name', 'Unknown')}")
                if len(st.session_state.mcp_tools) > 3:
                    st.write(f"... and {len(st.session_state.mcp_tools) - 3} more")

            try:
                if model_provider == "OpenAI":
                    try:
                        client = openai.OpenAI()
                        if st.session_state.mcp_tools:
                            response_stream = stream_openai_response_with_tools(
                                client, messages_for_api, model_name, st.session_state.mcp_tools
                            )
                        else:
                            response_stream = stream_openai_response(client, messages_for_api, model_name)
                    except Exception as openai_error:
                        st.error(f"OpenAI API Error: {str(openai_error)}")
                        if st.session_state.get('debug_mode', False):
                            st.error(f"Debug: Full error details: {openai_error}")
                        full_response = f"OpenAI API Error: {str(openai_error)}"
                        response_stream = iter([full_response])
                elif model_provider == "Anthropic":
                    client = anthropic.Anthropic()
                    if st.session_state.mcp_tools:
                        response_stream = stream_anthropic_response_with_tools(
                            client, messages_for_api, model_name, st.session_state.mcp_tools
                        )
                    else:
                        response_stream = stream_anthropic_response(client, messages_for_api, model_name)
                else:  # Google Gemini
                    genai.configure()
                    model = genai.GenerativeModel(model_name)
                    if st.session_state.mcp_tools:
                        response_stream = stream_gemini_response_with_tools(
                            model, messages_for_api, st.session_state.mcp_tools
                        )
                    else:
                        response_stream = stream_gemini_response(model, messages_for_api)
                
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "|")

                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                full_response = f"Error: {str(e)}"

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
    if st.sidebar.button("Finish"):
        os._exit(0)
