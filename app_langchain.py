import json
import asyncio

import anyio
from contextlib import asynccontextmanager
from typing import Tuple, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts.chat import ChatPromptTemplate


# MCP 서버 JSON 파일 경로
JSON_FILE = "my_mcp_servers.json"

# async def load_tools_from_json(json_file):
    # with open(json_file, "r", encoding="utf-8") as f:
    #     config = json.load(f)
    # servers = config["mcpServers"]

mcp_json_input = (
"""
{
  "mcpServers": {
    "gmail": {
      "command": "npx",
      "args": [
        "-y",
        "@gongrzhe/server-gmail-autoauth-mcp"
      ]
    },
    "google-search": {
      "command": "uvx",
      "args": ["mcp-google-cse"],
      "env": {
        "API_KEY": "AIzaSyCxYkilp1SyeGzAGvU4idjtxhzjfBnUGmM",
        "ENGINE_ID": "445477493f46346d6"
      }
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
"""
)

async def initialize_server(name: str, srv: dict) -> Tuple[Optional[list], Optional[tuple]]:
    client = None
    session = None
    result = (None, None)
    
    try:
        server_params = StdioServerParameters(
            command=srv["command"],
            args=srv.get("args", []),
            env=srv.get("env", {})
        )
        
        async with anyio.create_task_group() as tg:
            # 클라이언트 초기화
            client = stdio_client(server_params)
            read, write = await client.__aenter__()
            
            # 세션 초기화
            session = ClientSession(read, write)
            await session.initialize()
            
            # 도구 로딩
            print(f"'{name}' MCP 도구 로딩 중...")
            tools = await load_mcp_tools(session)
                
            print(f"'{name}' 서버 초기화 완료")
            result = (tools, (client, session))
        
    except* anyio.ExceptionGroup as e:
        print(f"'{name}' 서버 초기화 중 오류: {e.exceptions[0]}")
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"세션 정리 중 오류: {cleanup_error}")
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"클라이언트 정리 중 오류: {cleanup_error}")
    
    except* Exception as e:
        print(f"'{name}' 서버 초기화 중 오류: {e.exceptions[0]}")
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"세션 정리 중 오류: {cleanup_error}")
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"클라이언트 정리 중 오류: {cleanup_error}")
    
    return result

async def cleanup_resources(session, client):
    if session:
        try:
            await session.__aexit__(None, None, None)
        except Exception as e:
            print(f"세션 정리 중 오류: {e}")
    
    if client:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            print(f"클라이언트 정리 중 오류: {e}")

async def load_tools_from_json(mcp_json_input: str) -> Tuple[List, List]:
    print("도구 로딩 시작...")
    config = json.loads(mcp_json_input)
    servers = config["mcpServers"]

    all_tools = []
    all_sessions = []
    
    for name, srv in servers.items():
        print(f"서버 '{name}' 초기화 중...")
        tools, session_tuple = await initialize_server(name, srv)
        if tools and session_tuple:
            all_tools.extend(tools)
            all_sessions.append(session_tuple)
    
    return all_tools, all_sessions

async def main():
    sessions = []
    try:
        print("메인 함수 시작...")
        model = ChatOpenAI(model="gpt-4o")
        # tools, sessions = await load_tools_from_json(JSON_FILE)
        # 전체 도구 로딩 프로세스에 타임아웃 설정
        # 전체 프로세스 타임아웃 증가

        async with asyncio.timeout(180):  # 3분 타임아웃
            tools, sessions = await load_tools_from_json(mcp_json_input)
        
        if not tools:
            print("사용 가능한 도구가 없습니다.")
            return

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(model, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=3  # 최대 반복 횟수 제한
        )

        # invoke() 메서드는 async가 아닐 수 있으므로 수정
        try:
            result = agent_executor.invoke(
                {"input": "2024년 서울의 인구는"},
                return_only_outputs=True
            )
            print("결과:", result)
        except Exception as e:
            print(f"에이전트 실행 중 오류: {e}")

    except* asyncio.TimeoutError as e:
        print(f"전체 프로세스 시간 초과: {e.exceptions[0]}")
    except* Exception as e:
        print(f"오류 발생: {e.exceptions[0]}")
    finally:
        for client, session in sessions:
            await cleanup_resources(session, client)

    # 세션 정리
    # for client, session in sessions:
    #     await session.__aexit__(None, None, None)
    #     await client.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
