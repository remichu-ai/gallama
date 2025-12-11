import re
from typing import List, Tuple, Optional, Union
from ..data_classes.data_class import (
    TagDefinition
)
from abc import ABC, abstractmethod


class BaseStreamParser(ABC):
    @abstractmethod
    def process_stream(self, new_data: str) -> str:
        pass

    @abstractmethod
    def parse_full_text(self, text: str) -> List[Tuple[TagDefinition, str]]:
        pass

    @abstractmethod
    def flush(self) -> List[Tuple[TagDefinition, str]]:
        pass


class DummyParser(BaseStreamParser):
    def __init__(self, *args, **kwargs) -> None:
        self.default_tag_type = TagDefinition(
            tag_type="text",
            api_tag="content",
        )

    def process_stream(self, new_data: str) -> List[Tuple[TagDefinition, str]]:
        return [(self.default_tag_type, new_data)]

    def parse_full_text(self, text: str) -> List[Tuple[TagDefinition, str]]:
        return [(self.default_tag_type, text)]

    def flush(self) -> List[Tuple[TagDefinition, str]]:
        return []


class StreamParserByTag(BaseStreamParser):
    def __init__(
            self,
            tag_definitions: List[TagDefinition],
            default_tag_type=None,
            use_regex: bool = True
    ) -> None:

        self.buffer = ""
        self.stack: List[Tuple[TagDefinition, str]] = []
        self.tag_defs = tag_definitions
        self.use_regex = use_regex

        self.default_tag_type = default_tag_type or TagDefinition(
            tag_type="text",
            api_tag="content",
        )

        all_markers = [t.start_marker for t in tag_definitions if t.start_marker] + \
                      [t.end_marker for t in tag_definitions if t.end_marker]

        self.max_marker_length = max((len(m) for m in all_markers), default=1)

        self.start_tag_pattern = None
        if self.use_regex:
            pattern_parts = []
            for idx, td in enumerate(self.tag_defs):
                if td.start_marker:
                    pattern_parts.append(f'(?P<tag_{idx}>{re.escape(td.start_marker)})')
            if pattern_parts:
                self.start_tag_pattern = re.compile('|'.join(pattern_parts))

    def get_current_active_tag(self) -> TagDefinition:
        return self.stack[-1][0] if self.stack else self.default_tag_type

    def _find_start_tag_regex(self, text: str) -> Tuple[int, int, Optional[TagDefinition]]:
        if not self.start_tag_pattern:
            return -1, -1, None
        match = self.start_tag_pattern.search(text)
        if match:
            for name, val in match.groupdict().items():
                if val:
                    tag_idx = int(name.split('_')[1])
                    return match.start(), match.end(), self.tag_defs[tag_idx]
        return -1, -1, None

    def _find_start_tag_simple(self, text: str) -> Tuple[int, int, Optional[TagDefinition]]:
        best_idx = -1
        best_tag = None
        for td in self.tag_defs:
            if not td.start_marker: continue
            idx = text.find(td.start_marker)
            if idx != -1:
                if best_idx == -1 or idx < best_idx:
                    best_idx = idx
                    best_tag = td
                    if best_idx == 0: break
        if best_idx != -1 and best_tag:
            return best_idx, best_idx + len(best_tag.start_marker), best_tag
        return -1, -1, None

    def detect_start_tag(self, text: str) -> Tuple[int, int, Optional[TagDefinition]]:
        if self.use_regex:
            return self._find_start_tag_regex(text)
        return self._find_start_tag_simple(text)

    def push_tag_context(self, tag_identifier: Union[str, TagDefinition]) -> None:
        target_def = None
        if isinstance(tag_identifier, TagDefinition):
            target_def = tag_identifier
        else:
            for td in self.tag_defs:
                if td.tag_type == tag_identifier:
                    target_def = td
                    break
        if target_def:
            self.stack.append((target_def, target_def.end_marker))

    def clear_context(self) -> None:
        self.stack = []

    def process_stream(self, new_data: str) -> List[Tuple[TagDefinition, str]]:
        self.buffer += new_data
        results = []

        while True:
            current_closer = self.stack[-1][1] if self.stack else None
            current_tag = self.get_current_active_tag()

            start_found_idx, start_found_end, start_found_def = self.detect_start_tag(self.buffer)

            end_index = -1
            if current_closer:
                end_index = self.buffer.find(current_closer)

            # --- Logic Branching ---

            # Case A: Closing Tag Found
            if end_index != -1 and (start_found_idx == -1 or end_index < start_found_idx):
                content = self.buffer[:end_index]
                if content:
                    results.append((current_tag, content))

                # [NEW LOGIC] If configured, include the end marker in output
                if current_tag.include_markers:
                    results.append((current_tag, current_closer))

                self.buffer = self.buffer[end_index + len(current_closer):]
                self.stack.pop()
                continue

            # Case B: New Start Tag Found
            elif start_found_idx != -1:
                pre_content = self.buffer[:start_found_idx]
                # Emit text belonging to the *previous/outer* tag
                if pre_content:
                    results.append((current_tag, pre_content))

                if start_found_def:
                    # [NEW LOGIC] If configured, include the start marker in output.
                    # We capture specific slice [start:end] to handle regex variations if any.
                    if start_found_def.include_markers:
                        marker_text = self.buffer[start_found_idx:start_found_end]
                        results.append((start_found_def, marker_text))

                    self.stack.append((start_found_def, start_found_def.end_marker))

                self.buffer = self.buffer[start_found_end:]
                continue

            # Case C: Aggressive Streaming
            else:
                margin = self.max_marker_length
                if len(self.buffer) > margin:
                    chunk = self.buffer[:-margin]
                    results.append((current_tag, chunk))
                    self.buffer = self.buffer[-margin:]
                break

        return results

    def flush(self) -> List[Tuple[TagDefinition, str]]:
        results = []
        if self.buffer:
            current_tag = self.get_current_active_tag()
            results.append((current_tag, self.buffer))
            self.buffer = ""
        return results

    def parse_full_text(self, text: str, initial_tag: Union[str, TagDefinition] = None) -> List[
        Tuple[TagDefinition, str]]:
        # Reuse existing logic
        self.buffer = ""
        self.stack = []
        if initial_tag:
            self.push_tag_context(initial_tag)

        raw_results = self.process_stream(text)
        raw_results.extend(self.flush())
        raw_results = [_result for _result in raw_results if _result[1]]  # keep newlines

        merged_results: List[Tuple[TagDefinition, str]] = []
        for tag, content in raw_results:
            if merged_results and merged_results[-1][0].tag_type == tag.tag_type:
                prev_tag, prev_content = merged_results.pop()
                merged_results.append((prev_tag, prev_content + content))
            else:
                merged_results.append((tag, content))
        return merged_results