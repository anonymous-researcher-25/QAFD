from workflow.system_state import SystemState
from workflow.agents.tool import Tool

from llm.models import call_engine, get_llm_chain
from llm.prompts import get_prompt

class Agent:
    """
    Abstract base class for agents.
    """
    
    def __init__(self, name: str, task: str, config: dict):
        self.name = name
        self.task = task
        self.config = config
        self.tools_config = config["tools"]
        self.tools = {}
        self.chat_history = []
       
    
    def workout(self, system_state: SystemState) -> SystemState:
        """
        Abstract method to process the system state.

        Args:
            system_state (SystemState): The current system state.

        Returns:
            SystemState: The processed system state.
        """
        
        system_prompt = get_prompt(template_name="agent_prompt_v2")
        system_prompt = system_prompt.format(agent_name=self.name, 
                                             task=self.task, 
                                             tools=self.get_tools_description())
        self.chat_history.append({"role": "system", "content": system_prompt})

        # max number of tools in an agent
        ntools = 10
        
        
        if self.name=='Information Retriever':
            
            ntools = 3
            ordered_tools = ['extract_keywords', 'retrieve_entity', 'retrieve_context']

        if self.name=='schema_selector':
            if self.config["tools"]["select_tables"]['mode']=='corrects':
                ntools = 2                
            else:
                ntools = 3
                ordered_tools = ['filter_column', 'select_tables', 'select_columns']
        if self.name=="Candidate Generator":
            ntools = 2
            ordered_tools = ['generate_candidate', 'revise']

        
        try:
            for i in range(ntools):
                
                # response = self.call_agent(system_state) # choose a tool
                # tool_name = self.get_next_tool_name(response)
                tool_name = ordered_tools[i]
                print('.......', tool_name)
                self.chat_history.append({"role": "agent", "content": f"<tool_call>{tool_name}</tool_call>"})
                tool = self.tools[tool_name]
                try:
                    tool_response = self.call_tool(tool, system_state)
                    self.chat_history.append({"role": "tool_message", "content": tool_response})
                except Exception as e:
                    print(f"Error in tool {tool_name}: {e}")

        except Exception as e:
            print(f"Error in agent {self.name}: {e}")

 
        return system_state

    def call_tool(self, tool: Tool, system_state: SystemState) -> SystemState:
        """
        Call a tool with the given name and system state.
        """
        try:
            tool(system_state)
            return f"Tool {tool.tool_name} called successfully."
        except Exception as e:
            raise e
        
    def is_done(self, response: str) -> bool:
        """
        Check if the response indicates that the agent is done.
        """
        if "DONE" in response:
            return True
        return False
    
    def get_next_tool_name(self, response: str) -> str:
        """
        Get the next tool to call based on the response.
        """

        if self.config["engine"]=='meta-llama/Llama-3.3-70B':
            tool_name = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()

        elif self.config["engine"]=='deepseek':
            try:
                tool_name = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            except:
                if self.name=="Candidate Generator":
                    if "I'm the Candidate Generator agent" in response:
                        tool_name = 'generate_candidate' 

        else:
            tool_name = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return tool_name
    
    def call_agent(self, system_state: SystemState) -> SystemState:
        """
        Call the agent with the given system state.
        """
        
        messages = ""
        for chat in self.chat_history:
            role = chat["role"]
            content = chat["content"]
            messages += f"<{role}>\n{content}\n</{role}>\n"
        messages += f"<agent>\n"
        llm_chain = get_llm_chain(engine_name=self.config["engine"], temperature=0)

        response = call_engine(message=messages, engine=llm_chain)
        return response
        
    def get_tools_description(self) -> str:
        """
        Get the description of the tools.
        """
        tools_description = ""
        for i, tool in enumerate(self.tools):
            tools_description += f"{i+1}. {tool}\n"
        return tools_description
    
    def __call__(self, system_state: SystemState) -> SystemState:
        return self.workout(system_state)