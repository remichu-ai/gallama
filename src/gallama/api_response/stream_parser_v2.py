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
    def parse_full_text(
        self,
        text: str,
        initial_tag: Union[str, TagDefinition] = None,
    ) -> List[Tuple[TagDefinition, str]]:
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

    def parse_full_text(
        self,
        text: str,
        initial_tag: Union[str, TagDefinition] = None,
    ) -> List[Tuple[TagDefinition, str]]:
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
        self.generation_should_stop = False
        self.stop_reason: Optional[str] = None
        self._blocked_after_invalid_next_tag = False
        self._awaiting_allowed_next_tag: Optional[TagDefinition] = None

        self.default_tag_type = default_tag_type or TagDefinition(
            tag_type="text",
            api_tag="content",
        )

        all_markers = [t.start_marker for t in tag_definitions if t.start_marker] + \
                      [t.end_marker for t in tag_definitions if t.end_marker]
        start_markers = [t.start_marker for t in tag_definitions if t.start_marker]

        self.max_marker_length = max((len(m) for m in all_markers), default=1)
        self.max_start_marker_length = max((len(m) for m in start_markers), default=1)

        self.start_tag_pattern = None
        if self.use_regex:
            pattern_parts = []
            for idx, td in enumerate(self.tag_defs):
                if td.start_marker:
                    marker_pattern = td.start_marker
                    if td.marker_type != "regex":
                        marker_pattern = re.escape(marker_pattern)
                    pattern_parts.append(f'(?P<tag_{idx}>{marker_pattern})')
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
        best_end = -1
        best_tag = None
        for td in self.tag_defs:
            if not td.start_marker: continue
            if td.marker_type == "regex":
                match = re.search(td.start_marker, text)
                if match:
                    idx = match.start()
                    end = match.end()
                else:
                    idx = -1
                    end = -1
            else:
                idx = text.find(td.start_marker)
                end = idx + len(td.start_marker) if idx != -1 else -1
            if idx != -1:
                if best_idx == -1 or idx < best_idx:
                    best_idx = idx
                    best_end = end
                    best_tag = td
                    if best_idx == 0:
                        break
        if best_idx != -1 and best_tag:
            return best_idx, best_end, best_tag
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
        self._reset_post_tag_restrictions()

    def _reset_post_tag_restrictions(self) -> None:
        self.generation_should_stop = False
        self.stop_reason = None
        self._blocked_after_invalid_next_tag = False
        self._awaiting_allowed_next_tag = None

    def _resolve_allowed_next_tags(self, tag: TagDefinition) -> List[Tuple[str, Union[TagDefinition, str]]]:
        allowed_entries = tag.allowed_next_tag or []
        if not allowed_entries:
            return []

        resolved: List[Tuple[str, Union[TagDefinition, str]]] = []
        for entry in allowed_entries:
            if isinstance(entry, TagDefinition):
                resolved.append(("tag", entry))
                continue

            resolved.append(("literal", entry))

        return resolved

    def _activate_allowed_next_tag_guard(self, tag: TagDefinition) -> None:
        if tag.allowed_next_tag is not None:
            self._awaiting_allowed_next_tag = tag

    def _matches_allowed_next_tag_start(
        self,
        text: str,
        allowed_tags: List[Tuple[str, Union[TagDefinition, str]]],
    ) -> Optional[Tuple[str, Union[TagDefinition, str]]]:
        for entry_type, entry_value in allowed_tags:
            if entry_type == "literal":
                if text.startswith(entry_value):
                    return (entry_type, entry_value)
                continue

            tag = entry_value
            if not tag.start_marker:
                continue
            if tag.marker_type == "regex":
                match = re.match(tag.start_marker, text)
                if match and match.start() == 0:
                    return (entry_type, tag)
            elif text.startswith(tag.start_marker):
                return (entry_type, tag)

        return None

    def _could_be_allowed_next_tag_prefix(
        self,
        text: str,
        allowed_tags: List[Tuple[str, Union[TagDefinition, str]]],
    ) -> bool:
        if not text:
            return True

        for entry_type, entry_value in allowed_tags:
            if entry_type == "literal":
                if entry_value.startswith(text):
                    return True
                continue

            tag = entry_value
            if not tag.start_marker:
                continue
            if tag.marker_type == "string" and tag.start_marker.startswith(text):
                return True

            # Python's stdlib regex engine does not support partial matches.
            # Be conservative for regex-based markers and wait for a bit more data.
            if tag.marker_type == "regex" and len(text) < self.max_start_marker_length:
                return True

        return False

    def _apply_allowed_next_tag_guard(self, final: bool = False) -> str:
        if self._awaiting_allowed_next_tag is None:
            return "resume"

        guarded_tag = self._awaiting_allowed_next_tag
        allowed_tags = self._resolve_allowed_next_tags(guarded_tag)
        stripped_buffer = self.buffer.lstrip(" \t\r\n")

        if not stripped_buffer:
            if final:
                self.buffer = ""
                self._awaiting_allowed_next_tag = None
                return "resume"
            return "wait"

        allowed_match = self._matches_allowed_next_tag_start(stripped_buffer, allowed_tags)
        if allowed_match is not None:
            self.buffer = stripped_buffer
            self._awaiting_allowed_next_tag = None
            return "resume"

        if not allowed_tags:
            self.buffer = ""
            self._awaiting_allowed_next_tag = None
            self._blocked_after_invalid_next_tag = True
            self.generation_should_stop = True
            self.stop_reason = (
                f"Tag '{guarded_tag.tag_type}' must end the generation or be followed by EOF only"
            )
            return "stop"

        if not final and self._could_be_allowed_next_tag_prefix(stripped_buffer, allowed_tags):
            return "wait"

        self.buffer = ""
        self._awaiting_allowed_next_tag = None
        self._blocked_after_invalid_next_tag = True
        self.generation_should_stop = True
        self.stop_reason = (
            f"Tag '{guarded_tag.tag_type}' must be followed by EOF or one of {guarded_tag.allowed_next_tag}"
        )
        return "stop"

    def process_stream(self, new_data: str) -> List[Tuple[TagDefinition, str]]:
        if self._blocked_after_invalid_next_tag:
            return []

        self.buffer += new_data
        results = []

        while True:
            guard_action = self._apply_allowed_next_tag_guard(final=False)
            if guard_action == "wait":
                break
            if guard_action == "stop":
                break

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
                self._activate_allowed_next_tag_guard(current_tag)
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
        guard_action = self._apply_allowed_next_tag_guard(final=True)
        if guard_action == "stop":
            return results

        if self._blocked_after_invalid_next_tag:
            self.buffer = ""
            return results

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
        self._reset_post_tag_restrictions()
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
