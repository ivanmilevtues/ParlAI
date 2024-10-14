import os
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
import torch
from torch import nn
# from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_core.callbacks import CallbackManagerForToolRun

from pydantic.v1 import BaseModel, Field
from typing import List
from json.decoder import JSONDecodeError

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser

from langchain.agents import AgentType, initialize_agent, AgentExecutor, create_openai_functions_agent
from langchain_core.agents import AgentAction
from langchain_openai import ChatOpenAI


"""Util that calls Wikidata."""

from typing import List, Optional
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaQueryInput(BaseModel):
    """Input for the WikipediaQuery tool."""

    query: str = Field(description="query to look up on wikipedia")


BAD_DESCRIPTION = ("A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query.")

GOOD_DESCRIPTION = """A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
Question: The input should be the concrete event i.e. for: What is the significance of French Revolution in history? and What is the significance of 1789 Revolution in history? 
Query:French revolution
Question: How did Industrial Development impact City Growth?, How did Industrial Growth impact Urban Development?
Query: Industrialization and urbanization
"""

class WikipediaQueryRun(BaseTool):
    """Tool that searches the Wikipedia API."""

    name: str = "wikipedia"
    description: str = BAD_DESCRIPTION
    api_wrapper: WikipediaAPIWrapper
    total_runs = 0

    args_schema: Type[BaseModel] = WikipediaQueryInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool."""
        self.total_runs += 1
        return self.api_wrapper.run(query)


class OpenAIFuncFactory:
    @classmethod
    def create(cls, tool, model):
        try:
            agent = create_openai_functions_agent(model, [tool], hub.pull("hwchase17/openai-tools-agent"))
            agent = AgentExecutor(agent=agent, tools=[tool], verbose=True, return_intermediate_steps=True)
            agent.invoke({'input': "Can you call the available to you tool with random parameters?"})
            return agent
        except:
            return None


class ReactOldAgentFactory:

    @classmethod
    def create(cls, tool, model):
        try:
            agent = initialize_agent([tool], model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                                     return_intermediate_steps=True)
            agent.invoke({'input': "Can you call the available to you tool with random parameters?"})
            return agent
        except:
            return None


class ReactAgentOld:

    def __init__(self, tool, model):
        self.tool = tool
        self.agent = None
        agent_factories = [OpenAIFuncFactory, ReactOldAgentFactory]
        for factory in agent_factories:
            self.agent = factory.create(tool, model)
            if self.agent is not None:
                break

    def __call__(self, prompt, *args, **kwargs):
        agent_results = self.agent.invoke({'input': prompt})
        agent_answer = agent_results['output']
        tool_outputs = []
        for action, output in reversed(agent_results['intermediate_steps']):
            if isinstance(action, AgentAction) and action.tool == self.tool.name:
                tool_outputs.append(output)
        trace = self.get_trace(agent_results)
        return agent_answer, tool_outputs, trace, self.get_tool_arguments(agent_results)

    @staticmethod
    def get_trace(trace):
        """
        messages = [system("You are a helpful assistant. Your user is signed in as bob@mail.com"),
        user("Please do some research on Paris."),
        assistant(None, tool_call("1", "search_web", {"q": "bob@mail.com want's to know about Paris"})),
        tool("1", "Paris is the capital of France.")]
        """
        transformed_trace = [{'role': 'user', 'content': trace["input"]}]
        for message in trace['intermediate_steps']:
            if isinstance(message, tuple):
                if isinstance(message[0], AgentAction):
                    transformed_trace.append({'role': 'system', 'content': message[0].log})
                    transformed_trace.append({'role': 'assistant',
                                              'content': None,
                                              "tool_calls": [
                                                  {'id': 1,
                                                   'type': 'function',
                                                   'function': {'name': message[0].tool,
                                                                'arguments': message[0].tool_input}}
                                              ]})
                if isinstance(message[1], str):
                    transformed_trace.append({'role': 'tool', 'id': 1, 'content': message[1]})
        transformed_trace.append(f'assistant({trace["output"]}, None)')
        return transformed_trace

    @staticmethod
    def get_tool_arguments(trace):
        for message in trace['intermediate_steps']:
            if isinstance(message, tuple):
                if isinstance(message[0], AgentAction):
                    return message[0].tool_input
        return None

    def heartbeat(self) -> bool:
        return self.agent is not None

    def get_name(self) -> str:
        return 'react_old'

class AgentResult(BaseModel):
    ranking_result: List[int] = Field(description="The ranking results for the sentences, sentence one is at position 0, sentence two is at position 1 and so on.")

class ExampleBagOfWordsModel(nn.Module):
    """
    This constructs a simple bag of words model.

    It contains a encoder for encoding candidates and context.
    """

    def __init__(self, opt, dictionary):
        super().__init__()
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0)
        self.tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        self.agent = ReactAgentOld(self.tool, self.llm)
        self.tool.description = GOOD_DESCRIPTION
        print(f"This is with our description: {self.tool.description == GOOD_DESCRIPTION}")
        system_msg = "You are a helpful AI bot which is responsible for answering factual questions. You have access to the WikiTool which you can use to check factual data and no other tool. Usually I would advise you to search the relevant question in wikitools and then use the answer to rank the sentences"
        self.agent.agent = initialize_agent(tools=[self.tool], llm=self.llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                   system_message= system_msg,
                                   agent_kwargs={
                                       "system_message": system_msg},
                                   verbose=True,
                                   handle_parsing_errors="Check your output and make sure it conforms!",
                                   return_intermediate_steps=True, )
        self.parser = JsonOutputParser(pydantic_object=AgentResult)
        self.total_tool_usages = 0

    def encode_text(self, text_vecs):
        """
        This function encodes a text_vec to a text encoding.
        """
        return text_vecs

    def forward(self, observation, rerun=3):
        question = observation['text']
        answers = observation['label_candidates']
        if rerun == 0:
            return torch.zeros(1, len(answers))
        prompt = f"Given these {len(answers)} sentences: "
        for idx, a in enumerate(answers):
            prompt += f"\n{idx+1}. {a}"
        prompt += f'''
Can you give relevance score (10 means most relevant and 0 meaning least relevant) to each of the sentences according to the question: "{question}". Please check the question factuallity and then score.
{self.parser.get_format_instructions()}
'''
        try:
            agent_answer = self.llm.invoke([("human", prompt)]).content
            pre_call_tool_usage = self.tool.total_runs
            agent_answer, _, _, args = self.agent(prompt)
            if pre_call_tool_usage != self.tool.total_runs:
                self.total_tool_usages += 1
            result = self.parser.parse(agent_answer)
            rank_results = result['ranking_result']
            print(f"TOOL USAGES: {self.total_tool_usages}, {self.tool.total_runs}")
            return torch.tensor([rank_results], dtype=torch.float)
        except JSONDecodeError as e:
            return self.forward(observation, rerun-1)
        except OutputParserException:
            return self.forward(observation, rerun-1)
        except ValueError:
            return self.forward(observation, rerun-1)
        except KeyError:
            return self.forward(observation, rerun-1)


class LangchainAgent(TorchRankerAgent):
    """
    Example subclass of TorchRankerAgent.

    This particular implementation is a simple bag-of-words model, which demonstrates
    the minimum implementation requirements to make a new ranking model.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add CLI args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        arg_group = parser.add_argument_group('ExampleBagOfWordsModel Arguments')
        arg_group.add_argument('--hidden-dim', type=int, default=512)
        return parser

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        This function takes in a Batch object as well as a Tensor of candidate vectors.

        It must return a list of scores corresponding to the likelihood that the
        candidate vector at that index is the proper response. If `cand_encs` is not
        None (when we cache the encoding of the candidate vectors), you may use these
        instead of calling self.model on `cand_vecs`.
        """
        scores = self.model.forward(self.observation)
        return scores

    def build_model(self):
        """
        This function is required to build the model and assign to the object
        `self.model`.
        """
        return ExampleBagOfWordsModel(self.opt, self.dict)
