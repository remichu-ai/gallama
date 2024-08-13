import re
from typing import List, Tuple, Optional, Union
from gallama.data_classes.data_class import TextTag, ArtifactTag


class StreamParser:
    def __init__(self):
        self.buffer = ""
        self.current_element = None
        self.current_tag = None
        self.current_content = ""
        self.MALFORMED_CHECK_LENGTH = 300
        self.in_answer_tag = False
        # self.xml_prefix = "```xml\n"
        self.xml_prefix = ""
        self.root_key = "<answer>"
        self.full_root_key = f"{self.xml_prefix}{self.root_key}"
        self.tag_pattern = re.compile(r'(<artifact\s+(?:(?:identifier|type|title|language)="[^"]*"\s*)*>)|(<text>)')
        self.comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)

    def process_stream(self, new_data: str) -> List[Tuple[Union[TextTag, ArtifactTag], str]]:
        self.buffer += new_data
        self.remove_comments()

        results = []

        while True:
            if not self.in_answer_tag:
                start_index = self.buffer.find(self.full_root_key)
                if start_index != -1:
                    self.buffer = self.buffer[start_index + len(self.full_root_key):]
                    self.in_answer_tag = True
                elif len(self.buffer) >= self.MALFORMED_CHECK_LENGTH:
                    results.append((TextTag(), self.buffer))
                    self.buffer = ""
                else:
                    break

            if self.current_element is None:
                match = self.tag_pattern.search(self.buffer)
                if match:
                    element_type = "artifact" if match.group(1) else "text"
                    start_index = match.start()
                    tag_length = match.end() - match.start()

                    self.current_element = element_type
                    if element_type == "artifact":
                        self.current_tag = self._parse_artifact_tag(self.buffer[start_index:match.end()])
                    else:
                        self.current_tag = TextTag()

                    self.buffer = self.buffer[start_index + tag_length:]
                    self.current_content = ""
                elif len(self.buffer) >= self.MALFORMED_CHECK_LENGTH:
                    results.append((TextTag(), self.buffer))
                    self.buffer = ""
                else:
                    break
            elif len(self.buffer) <= len(f'</{self.current_element}>'):
                break
            else:
                end_tag = f'</{self.current_element}>'
                end_index = self.buffer.find(end_tag)
                if end_index != -1:
                    content = self.buffer[:end_index]
                    results.append((self.current_tag, content))
                    self.buffer = self.buffer[end_index + len(end_tag):]
                    self.current_element = None
                    self.current_tag = None
                    self.current_content = ""
                else:
                    if self.buffer:
                        content_chunk = self.buffer[:len(self.buffer) - len(end_tag)]
                        self.buffer = self.buffer[len(self.buffer) - len(end_tag):]
                        results.append((self.current_tag, content_chunk))
                    break

        return results

    def remove_comments(self):
        comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)
        self.buffer = comment_pattern.sub('', self.buffer)

    def _parse_artifact_tag(self, tag_content: str) -> ArtifactTag:
        attributes = re.findall(r'(\w+)="([^"]*)"', tag_content)
        attr_dict = dict(attributes)
        return ArtifactTag(
            artifact_type=attr_dict.get('type', 'code'),
            identifier=attr_dict.get('identifier', ''),
            title=attr_dict.get('title', ''),
            language=attr_dict.get('language')
        )

    def get_current_state(self) -> Tuple[Optional[str], str]:
        return self.current_element, self.current_content

    def parse_full_response(self, response: str) -> List[Tuple[Union[TextTag, ArtifactTag], str]]:
        if not response.startswith(self.xml_prefix):
            response = f"{self.xml_prefix}{response}"
        response = self.comment_pattern.sub('', response)
        return self.process_stream(response)