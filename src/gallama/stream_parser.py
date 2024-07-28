import re
from typing import List, Tuple, Optional, Literal


class StreamParser:
    def __init__(self, quote_type: str = "'''"):
        self.quote_type = quote_type
        self.object_start_pattern = re.compile(r'\s*{\s*"content_type"\s*:\s*"(\w+)"\s*,\s*"content"\s*:\s*r' + quote_type)
        self.content_end_pattern = re.compile(quote_type + r'\s*}\s*,?\s*')
        self.buffer = ""
        self.current_content_type = None
        self.current_content = ""

    def process_stream(self, new_data: str) -> List[Tuple[Optional[Literal['text', 'code']], str]]:
        self.buffer += new_data
        results = []
        last_processed_index = 0  # Track the end of the last processed content

        while True:
            if self.current_content_type is None:
                match = self.object_start_pattern.search(self.buffer, last_processed_index)
                if match:
                    self.current_content_type = match.group(1)
                    content_start = self.buffer.index(self.quote_type, match.start()) + len(self.quote_type)
                    last_processed_index = content_start
                    self.current_content = ""
                else:
                    break
            else:
                end_match = self.content_end_pattern.search(self.buffer, last_processed_index)
                if end_match:
                    content_end = end_match.start()
                    self.current_content += self.buffer[last_processed_index:content_end]
                    results.append((self.current_content_type, self.current_content))
                    last_processed_index = end_match.end()
                    self.current_content_type = None
                    self.current_content = ""
                else:
                    # Append incremental content to current_content
                    self.current_content += self.buffer[last_processed_index:]
                    results.append((self.current_content_type, self.buffer[last_processed_index:]))
                    last_processed_index = len(self.buffer)
                    break

        self.buffer = self.buffer[last_processed_index:]
        return results

    def get_current_state(self) -> Tuple[Optional[str], str]:
        return self.current_content_type, self.current_content

    def parse_full_response(self, response: str) -> List[Tuple[Optional[Literal['text', 'code']], str]]:
        self.buffer = response
        results = []
        last_processed_index = 0

        while last_processed_index < len(self.buffer):
            match = self.object_start_pattern.search(self.buffer, last_processed_index)
            if match:
                content_type = match.group(1)
                content_start = self.buffer.index(self.quote_type, match.start()) + len(self.quote_type)
                end_match = self.content_end_pattern.search(self.buffer, content_start)
                if end_match:
                    content_end = end_match.start()
                    content = self.buffer[content_start:content_end]
                    results.append((content_type, content))
                    last_processed_index = end_match.end()
                else:
                    break
            else:
                break

        return results