import json
import yaml
from functools import lru_cache
import copy
from typing import List, Dict, Union, Literal
from gallama.data_classes.data_class import (
    BaseMessage,
    ChatMLQuery,
    ToolCall,
    MultiModalTextContent,
    MultiModalImageContent,
    MultiModalImageHFContent,
    MultiModalAudioContent, TagDefinition
)
from gallama.logger.logger import logger
from pydantic import BaseModel
from copy import deepcopy
from gallama.utils.utils import parse_xml_to_dict
from fastapi import HTTPException
from pathlib import Path
from textwrap import dedent
import uuid
from transformers import AutoTokenizer, AutoConfig
from .model_special_tag import MODEL_SPECIAL_TAG, MODEL_EOS_TOKEN, MODEL_VISION_TOKEN
from ....api_response.stream_parser_v2 import StreamParserByTag





class PromptEngineTransformers:
    def __init__(self, prompt_format: str| None = None, model_path: str = None):
        logger.info("Use transformers tokenizer for prompt templating")

        assert model_path is not None

        self._transformer_tokenizer = AutoTokenizer.from_pretrained(model_path)
        # patch the template to standardized format
        self.patch_thinking_template()

        self._transformer_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model_prompt = None
        self.model_type = self._transformer_config.model_type
        logger.info(f"transformer model_type: {self.model_type}")

        self.special_tag = MODEL_SPECIAL_TAG.get(self.model_type, None)
        self._tag_definitions = list(self.special_tag.values()) if self.special_tag else None
        self.thinking_tag = self.special_tag.get("thinking") if self.special_tag else None

        if self._tag_definitions:
            self.tag_parser = StreamParserByTag(
                tag_definitions=self._tag_definitions,
                default_tag_type=None
            )
        else:
            self.tag_parser = None

        self.system_msg_enabled = True  # TODO check jinja template?
        self.tool_enabled = True    # TODO check jinja template?

        # set eos token
        self.eos_token_list = [self._transformer_tokenizer.special_tokens_map.get("eos_token")]
        # check if there is custom eos token
        if MODEL_EOS_TOKEN.get(self.model_type, None) is not None:
            self.eos_token_list.extend(MODEL_EOS_TOKEN[self.model_type])

        self._vision_token = MODEL_VISION_TOKEN.get(self.model_type, None)

        # make unique, transformers might return None hence need to exclude
        self.eos_token_list = list(set([_s for _s in self.eos_token_list if _s is not None]))

        self.is_thinking_model = self.check_thinking_model()
        self.support_list_content = self._template_supports_list_content(self._transformer_tokenizer)
        self.support_developer_role = self._template_supports_developer_role(self._transformer_tokenizer)



    @property
    def tag_definitions(self):
        return self._tag_definitions

    @property
    def tag_dict(self):
        return self.special_tag

    @property
    def is_thinking(self):
        return self.is_thinking_model

    @property
    def vision_token(self):
        return self._vision_token

    def check_thinking_model(self):
        """
        Checks if the loaded tokenizer's chat_template contains logic
        for reasoning or reasoning_content.
        """
        template = self._transformer_tokenizer.chat_template
        if not template:
            return False

        # Check for the standard HF field or your custom field
        return "message.reasoning_content" in template or "message.reasoning" in template or "reasoning_content" in template

    def patch_thinking_template(self):
        """
        Patches the tokenizer's chat_template to use 'message.reasoning'
        instead of 'message.reasoning_content'.
        """
        if not self._transformer_tokenizer.chat_template:

            return

        # Check if the standard HF field is present
        if "message.reasoning_content" in self._transformer_tokenizer.chat_template:
            # Replace the Jinja variable access
            self._transformer_tokenizer.chat_template = self._transformer_tokenizer.chat_template.replace(
                "message.reasoning_content",
                "message.reasoning"
            )
            logger.info(f"Patched chat template: reasoning_content -> reasoning")
        else:
            logger.info("No patching needed or target variable not found.")

    @lru_cache(maxsize=1)
    def _template_supports_list_content(self, tokenizer) -> bool:
        """
        Probes the tokenizer's chat template to see if it handles list-based content correctly.

        It sends: [{"role": "user", "content": [{"type": "text", "text": "PROBE_TEST"}]}]

        If the template returns "PROBE_TEST", it supports lists.
        If the template returns "[{'type': 'text', 'text': 'PROBE_TEST'}]", it only supports strings.
        """
        # 1. Create a safe dummy message
        probe_msg = [
            {"role": "user", "content": [{"type": "text", "text": "PROBE_TEST_TOKEN"}]}
        ]

        try:
            # 2. Render it (tokenize=False gives us the raw string)
            result = tokenizer.apply_chat_template(probe_msg, tokenize=False)

            # 3. Check the output
            # If the raw list structure appears in the output, the template failed to parse it.
            if "[{'type': 'text'" in result or '[{"type": "text"' in result:
                logger.info("Chat template does not support list content")
                return False

            # If our probe token exists but the list syntax doesn't, it worked.
            if "PROBE_TEST_TOKEN" in result:
                logger.info("Chat template supports list content")
                return True

        except Exception as e:
            # If the template crashes on a list input, it definitely doesn't support it.
            logger.debug("Failed to probe whether list content is supported -> set to not supported")
            return False

        return False

    @lru_cache(maxsize=1)
    def _template_supports_developer_role(self, tokenizer) -> bool:
        """
        Probes the tokenizer's chat template to see if it supports the 'developer' role.
        """
        probe_msg = [{"role": "developer", "content": "PROBE_TEST_DEVELOPER"}]
        try:
            result = tokenizer.apply_chat_template(probe_msg, tokenize=False)
            if "PROBE_TEST_DEVELOPER" in result:
                return True
        except Exception:
            # Template throws a Jinja error or ValueError if the role is unsupported
            return False

        return False

    def convert_openai_to_hf_format(self, messages: List[Dict]) -> List[Dict]:
        """
        Converts OpenAI Chat Format to Hugging Face transformers format.

        1. Converts {"type": "image_url"} -> {"type": "image"}
        2. Preserves all other content types (text, etc.) exactly as is.
        3. Preserves all top-level keys (tool_calls, tool_call_id, name, etc.)
        """
        hf_messages = []

        for msg in messages:
            # 1. Deep copy ensures we keep 'tool_calls', 'tool_call_id', 'role', etc.
            #    without mutating the original input.
            new_msg = copy.deepcopy(msg)

            content = msg.get("content")

            # 2. Handle List Content (Multimodal or Structured Text)
            if isinstance(content, list):
                new_content = []
                for item in content:
                    item_type = item.get("type")

                    # TRANSFORM: OpenAI Image Format -> HF Image Format
                    if item_type == "image_url":
                        image_data = item.get("image_url", {})
                        url = image_data.get("url")
                        if url:
                            new_content.append({
                                "type": "image",
                                "image": url
                            })

                    # PASS THROUGH: Text, standard items, or custom types
                    else:
                        new_content.append(item)

                new_msg["content"] = new_content

            # 3. Handle String Content (Standard Text or Tool Results)
            #    We leave these exactly as is.

            hf_messages.append(new_msg)

        return hf_messages

    def get_prompt(
        self,
        query: ChatMLQuery,
        pydantic_tool_dict: List[BaseModel] = None,
        answer_format_schema: bool = True,  # whether to add the instruction for tool calling answer schema
        prefix_prompt: str = "",
        leading_prompt: str = "",
        thinking_template: str = None,
        thinking_response: str = None,
        backend: Literal["exllama", "llama_cpp", "transformers", "embedding"] = "exllama",    # skip model pseudo token and use exllama placeholder token # TODO - code refractoring
        pydantic_tool_code: str = None,     # the code representation of tool
    ) -> str:

        # 1. Basic dump of messages
        conversation = [_m.model_dump() for _m in query.messages]

        # 2. Convert 'developer' to 'system' if the template doesn't support 'developer'
        if not self.support_developer_role:
            for msg in conversation:
                if msg.get("role") == "developer":
                    msg["role"] = "system"

        # patch image format
        conversation = self.convert_openai_to_hf_format(conversation)

        # Before passing messages to tokenizer.apply_chat_template:
        # convert tool call to json string
        for msg in conversation:
            if msg.get("tool_calls"):
                for tool in msg["tool_calls"]:
                    if isinstance(tool["function"]["arguments"], str):
                        tool["function"]["arguments"] = json.loads(tool["function"]["arguments"])

        # 3. If template is "Dumb" (Text-Only), we MUST flatten text lists.
        #    If template is "Smart" (Multimodal), we leave it alone.
        if not self.support_list_content:

            for index, _m in enumerate(conversation):
                _content = _m.get("content")

                # Check if this specific message is a list
                if isinstance(_content, list):

                    # Safety Check: We can only flatten if it is PURELY text.
                    # If there are images in a text-only template, we can't do anything helpful anyway.
                    is_pure_text = all((c.get("type") == "text") for c in _content)

                    if is_pure_text:
                        # Join all text chunks
                        _content_str = "".join([c.get("text", "") for c in _content])
                        conversation[index]["content"] = _content_str

        enable_thinking = query.reasoning_effort is not None

        # 4. Get prompt from transformers
        prompt = self._transformer_tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            # add_generation_prompt=conversation[-1]["role"] == "user",
            add_generation_prompt=True,
            tools=[_t.model_dump() for _t in query.tools] if query.tools else None,
            enable_thinking=enable_thinking
        )

        # handling thinking
        starting_tag = TagDefinition(tag_type="text")

        # Debugging logs (Optional: cleanup if not needed)
        logger.info(f"Checking thinking model: {self.is_thinking_model}")
        if self.thinking_tag:
            logger.info(f"Thinking tag start: {self.thinking_tag.start_marker}")

        if self.is_thinking_model and self.thinking_tag:
            # Handle explicit reasoning effort cleanup (existing logic)
            if query.reasoning_effort is None and self.thinking_tag and self.thinking_tag.end_marker:
                # Logic for when reasoning is explicitly disabled/handled via prompt injection
                # (This line from your original code seemed to force close a tag)
                prompt += f"\n{self.thinking_tag.end_marker}"

            else:
                # Check for OPEN thinking tag anywhere in the prompt
                s_marker = self.thinking_tag.start_marker.strip()
                e_marker = self.thinking_tag.end_marker.strip() if self.thinking_tag.end_marker else ""

                # Find the position of the *last* occurrence of start and end markers
                last_start_idx = prompt.rfind(s_marker)
                last_end_idx = prompt.rfind(e_marker) if e_marker else -1

                # Determine if the tag is currently open:
                # 1. We found a start marker (idx != -1)
                # 2. AND (We found no end marker OR the start marker appears AFTER the last end marker)
                is_tag_open = last_start_idx != -1 and (
                        last_end_idx == -1 or last_start_idx > last_end_idx
                )

                if is_tag_open:
                    logger.info("Prompt contains an open thinking tag (start > end)")
                    # Let streaming know that we are starting inside a thought block
                    starting_tag = self.thinking_tag

        return prompt, starting_tag
