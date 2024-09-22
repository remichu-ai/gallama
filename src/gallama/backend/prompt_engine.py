import json
import yaml
from typing import List, Dict
from gallama.data_classes.data_class import BaseMessage, ChatMLQuery, ToolCall
from pydantic import BaseModel
from copy import deepcopy
from gallama.utils.utils import parse_xml_to_dict
from fastapi import HTTPException
from pathlib import Path
from textwrap import dedent
from ..data import ARTIFACT_SYSTEM_PROMPT


class PromptEngine:
    def __init__(self, prompt_format: str):
        self.system_msg_enabled = False
        self.tool_enabled = False
        self.model_prompt_all = self.get_prompt_token()
        if not self.model_prompt_all.get(prompt_format):
            raise ValueError(f'Prompt format {prompt_format} not found in data/model_token.yaml')

        self.model_prompt = self.model_prompt_all.get(prompt_format)
        self.system_msg_enabled = self.model_prompt.get("system_msg_enabled")
        self.tool_enabled = self.model_prompt.get("tool_enabled")
        self.eos_token_list = self.model_prompt.get("eos_token_list")

    @staticmethod
    def get_prompt_token() -> Dict:
        """Get the absolute path to the data directory."""
        yaml_file = Path(__file__).parent.parent / 'data' / 'model_token.yaml'
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data

    def get_conversation_start_token(self):
        return self.model_prompt.get("conversation_start", "")

    def get_conversation_end_token(self):
        return self.model_prompt.get("conversation_end", "")

    def leading_prompt_token(self):
        return self.model_prompt.get("leading_prompt_token", "")

    def get_user_start_token(self):
        return self.model_prompt.get("user_start", "")

    def get_user_end_token(self):
        return self.model_prompt.get("user_end", "")

    def get_sys_start_token(self):
        return self.model_prompt.get("system_start", "")

    def get_sys_end_token(self):
        return self.model_prompt.get("system_end", "")

    def get_assistant_start_token(self):
        return self.model_prompt.get("assistant_start", "")

    def get_assistant_end_token(self):
        return self.model_prompt.get("assistant_end", "")

    def get_tool_start_token(self):
        return self.model_prompt.get("tool_start", "")

    def get_tool_end_token(self):
        return self.model_prompt.get("tool_end", "")

    def get_tool_result_start_token(self):
        return self.model_prompt.get("tool_result_start", "")

    def get_tool_result_end_token(self):
        return self.model_prompt.get("tool_result_end", "")

    def get_tool_call_start_token(self):
        return self.model_prompt.get("tool_call_start", "")

    def get_tool_call_end_token(self):
        return self.model_prompt.get("tool_call_start", "")

    def _get_role_token(self, role, token_type: str):
        if token_type == "start":
            if role == "system":
                return self.get_sys_start_token()
            elif role == "user":
                return self.get_user_start_token()
            elif role == "assistant":
                return self.get_assistant_start_token()
            elif role == "tool":
                return self.get_tool_start_token()
            elif role == "tool_result":
                return self.get_tool_result_start_token()
            elif role == "tool_call":
                return self.get_tool_call_start_token()
        elif token_type == "end":
            if role == "system":
                return self.get_sys_end_token()
            elif role == "user":
                return self.get_user_end_token()
            elif role == "assistant":
                return self.get_assistant_end_token()
            elif role == "tool":
                return self.get_tool_end_token()
            elif role == "tool_result":
                return self.get_tool_result_end_token()
            elif role == "tool_call":
                return self.get_tool_call_end_token()
        else:
            return ""

    @staticmethod
    def _get_message_type(msg: BaseMessage) -> str:
        if msg.tool_call_id:
            return "tool_result"
        elif msg.role == "tool":
            return "tool_call"
        elif msg.role == "assistant":
            return "assistant"
        elif msg.role == "system":
            return "system"
        elif msg.role == "user":
            return "user"
        else:
            raise ValueError(f"Unknown message type {msg.role}")

    def _format_tool_call(self, tool_call: ToolCall) -> str:
        """
            get string representation of tool call like normal python function calling
            Example Output: get_current_weather(location='Boston', unit='Fahrenheit')
        """

        func_name = tool_call.function.name
        args_str = tool_call.function.arguments.replace('"', '')
        return f"{func_name}({args_str})"

    def _format_tool_result(self, msg: BaseMessage) -> str:
        content = msg.content if msg.content else ""

        try:
            content = f"Result of tool call reference id {msg.tool_call_id}:\n" + str(json.dumps(json.loads(content), indent=2)) + "\n\n"
        except:
            content = f"Result of tool call reference id {msg.tool_call_id}:\n" + str(content) + "\n\n"

        return content

    def _format_tool_call_msg(self, msg: BaseMessage) -> str:
        """one msg might have multiple tool calls"""
        tool_call_str = f"Please help to call these tool:\n"

        for tool_call in msg.tool_calls:
            tool_call_str += f"Request for tool call with reference id {tool_call.id}:\n"
            tool_call_str += self._format_tool_call(tool_call)
            tool_call_str += "\n\n"

        return tool_call_str

    def _format_tool_msg(self, pydantic_tool_list: Dict[str, BaseModel]) -> str:
        # append tools calling if it is a prompt
        tools_json = ("\nBelow are the functions available to you to use.\n"
                      "If you need to use multiple functions, please provide it in chronological order.\n"
                      "If user or system prompt request certain restriction (examples: can only use one tool, "
                      "must use a specific tool etc; then please respect the instruction.)\n\n")

        if self.tool_enabled:
            tools_json += self.get_tool_start_token()

        for tool_name, tool in pydantic_tool_list.items():
            tools_json = tools_json + tool_name + ":\n" + str(json.dumps(tool.schema(), indent=2)) + "\n---\n"

        if self.tool_enabled:
            tools_json += self.get_tool_end_token()

        return tools_json

    def _get_one_msg(self, msg: BaseMessage) -> str:
        content = msg.content if msg.content else ""

        prompt = ""
        tool_call_str = ""

        if self._get_message_type(msg) == "tool_result":
            return self._format_tool_result(msg)
        elif self._get_message_type(msg) == "tool_call":
            return self._format_tool_call_msg(msg)
        elif self._get_message_type(msg) in ["assistant", "system", "user"]:
            return content

    def _get_one_msg_grp(self, grp: List[BaseMessage]) -> str:
        prompt = self._get_role_token(role=self._get_message_type(grp[0]), token_type="start")

        for msg in grp:
            prompt = prompt + self._get_one_msg(msg) + "\n\n"

        prompt += self._get_role_token(role=self._get_message_type(grp[0]), token_type="end")
        return prompt

    def _regroup_msg(self, msg_List: List[BaseMessage]):
        # group msg with same role into one array
        regrouped_msg = []
        temp_array = []

        # first create a list of msg with the role converted
        for idx, msg in enumerate(msg_List):
            temp_msg = deepcopy(msg)
            if temp_msg.role == "tool" and not self.tool_enabled:
                temp_msg.role = "user"      # convert to user as this LLM model does not have tool role
            elif temp_msg.role == "system" and not self.system_msg_enabled:
                temp_msg.role = "user"  # convert to user as this LLM model does not have system role

            temp_array.append(temp_msg)

        # now grouping message that was from the same role together
        # this is cause model might have been trained with alternated role prompting
        # and repeat of the same role will not give good response
        previous_msg = msg_List[0]
        grouped_array = []
        for idx, msg in enumerate(temp_array):
            if msg.role == previous_msg.role:
                grouped_array.append(msg)
            else:
                if grouped_array:
                    # create new array
                    regrouped_msg.append(grouped_array.copy())
                    grouped_array = [msg]     # reset temp array
                else:
                    grouped_array.append(msg)

            previous_msg = msg

        # add the last group
        if temp_array:
            regrouped_msg.append(grouped_array.copy())

        # logger.debug("overall regroup:\n" + str(regrouped_msg))
        return regrouped_msg

    def get_prompt(
        self,
        query: ChatMLQuery,
        pydantic_tool_dict: List[BaseModel] = None,
        answer_format_schema: bool = True,  # whether to add the instruction for tool calling answer schema
        leading_prompt: str = None,
        use_thinking: bool = True,
        thinking_template: str = None,
        thinking_response: str = None,
        #prompt_mode: Literal["auto"]
    ) -> str:

        def _create_leading_prompt(original_prompt, query_leading_prompt, model_leading_prompt):
            # add leading prompt from user or model
            # this is usually the role e.g. assistant:, Tom to Jerry:
            _leading_prompt = ""
            if query_leading_prompt:
                _leading_prompt = model_leading_prompt + query_leading_prompt
            elif model_leading_prompt:
                _leading_prompt = model_leading_prompt

            if not _leading_prompt.endswith("\n"):
                _leading_prompt += "\n"

            return _leading_prompt

        thinking_example = ""

        # thinking_example = dedent("""
        # ### Thinking example:
        # Question: Is dog or cat faster? Answer in Capital letter
        # Apply thinking template:
        # <format_restriction>Any specific format requirement from user</format_restriction>
        # My thought process using the thinking template:
        # <format_restriction>User request for answer in capital letter</format_restriction>
        # Final answer:
        # CAT
        # End of Thinking Example.
        # """).strip()


        prompt = self.get_conversation_start_token()     # use arrange to story prompt

        # add the default system prompt for artifact
        if query.artifact != "No":
            prompt += ARTIFACT_SYSTEM_PROMPT

        msg_groups = self._regroup_msg(query.messages)

        # concat all msg
        for idx, msg_group in enumerate(msg_groups):
            prompt = prompt + self._get_one_msg_grp(msg_group)

        # if there is tool, return the prompt with tool
        if pydantic_tool_dict and query.tool_choice != "none":
            prompt = prompt + self._format_tool_msg(pydantic_tool_dict)

            if answer_format_schema:
                prompt += """
IMPORTANT: If you use tool/ functions, please answer using the following schema using json format.
Each item in the array under "functions_calling" is the "name" of the function to call and its "arguments".

{
  "functions_calling": [
    {
      "name": the name of function/ tool to call,
      "arguments": {argument_name: argument_value}
    }
}


If no functions_calling/ tool needed, the response can be:
{
  "functions_calling": []
}
End of Example of answer with Tool/ Function_calling usage.

"""
        # TODO move the thinking_template check to request receiver
        # initialize thinking_template_dict
        thinking_template_dict = {}
        root_key = None
        if thinking_template:
            try:
                thinking_template_dict = parse_xml_to_dict(thinking_template)
                if len(list(thinking_template_dict.keys())) > 1:
                    raise HTTPException(status_code=400, detail=f"thinking_template must be a XML with only 1 root key")

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"thinking_template is not valid XML string")

            # find the XML root key
            root_key = list(thinking_template_dict.keys())[0]

        # prompt creation with thinking_template
        if use_thinking and thinking_template and not thinking_response:
            prompt += _create_leading_prompt(prompt, query.leading_prompt, self.leading_prompt_token())

            prompt += "\nNow, before answering the question, I am required to apply XML thinking template to guide my internal thinking process.\n" + \
                      f"{thinking_example}\n" + \
                      "The thinking template i need to apply to answer this question is as follow:\n" + \
                      thinking_template + "\n" + \
                      f"My thinking using the above XML template as follow:\n\n```xml\n"
            # add ending token
            # prompt += self.get_conversation_end_token()

        elif use_thinking and thinking_template and thinking_response:
            # the thinking result is ready here
            # root key need to be added as we use it for leading prompt in the prompt creation above so it is not part of the response
            prompt += "\nNow, before answering the question, I am required to apply XML thinking template to guide my internal thinking process.\n" + \
                      f"{thinking_example}\n" + \
                      "Now, the thinking template i need to apply to answer this question is:\n" + \
                      thinking_template + "\n" + \
                      f"My thinking using the XML template as follow:\n```xml\n{thinking_response}\n" + \
                      "Now answer the question. Remember that the thinking above is INVISIBLE to user, " + \
                      "and you are PROHIBITED to mentioned to user about the existence of Thinking section. You are ALLOWED to reiterate points from thinking section to user if necessary."

            # add ending token
            prompt += self.get_conversation_end_token()

            # for thinking response, we put it before the leading prompt
            prompt += _create_leading_prompt(prompt, query.leading_prompt, self.leading_prompt_token())

        else:
            # add ending token
            prompt += self.get_conversation_end_token()

            prompt += _create_leading_prompt(prompt, query.leading_prompt, self.leading_prompt_token())


        # add leading_prompt from code which is usually tool related
        # e.g. ```json
        if leading_prompt:      # leading prompt that passed to this function take highest priority
            prompt += leading_prompt + "\n"

        return prompt

        # match tool call result #TODO

