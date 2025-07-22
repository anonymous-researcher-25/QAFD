from typing import Dict

from llm.models import get_llm_chain, async_llm_chain_call
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class ExtractKeywords(Tool):
    """
    Tool for extracting keywords from the question and hint.
    """
    
    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()
        
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name

        # prompt=get_prompt(template_name=self.template_name)
        # engine=get_llm_chain(**self.engine_config)
        # = { "QUESTION": "what is the capital of france", "HINT":"None"} 
        # chain = prompt | engine
        # prompt_text = prompt.invoke(request_kwargs).messages[0].content
        # prompt_text = prompt.invoke(request_list[0]).messages[0].content
        # output = chain.invoke(request_kwargs)   chain.invoke(request_list[0])
        
    def _run(self, state: SystemState):
        request_kwargs = { "QUESTION": state.task.question, "HINT": state.task.evidence,}

        response = async_llm_chain_call(
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_list=[request_kwargs],
            step=self.tool_name,
            sampling_count=1
        )[0]
        state.keywords = response[0]

    def _get_updates(self, state: SystemState) -> Dict:
        return {"keywords": state.keywords}