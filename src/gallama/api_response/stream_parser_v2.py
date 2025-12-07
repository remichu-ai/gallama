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
        pass

    def process_stream(self, new_data: str) -> str:
        return new_data

    def parse_full_text(self, text: str) -> str:
        return text

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
        self.MALFORMED_CHECK_LENGTH = 300
        self.tag_defs = tag_definitions
        self.use_regex = use_regex

        self.default_tag_type = default_tag_type or TagDefinition(
            tag_type="text",
            api_tag="content",
        )

        # Calculate max marker length for tail protection
        # We need to consider both start and end markers because we might be
        # waiting for a closing tag (e.g. </thought>) to appear.
        all_markers = [t.start_marker for t in tag_definitions] + [t.end_marker for t in tag_definitions]

        # This determines the maximum characters we hold back
        self.max_marker_length = max((len(m) for m in all_markers), default=1)

        # Build Regex only if enabled
        self.start_tag_pattern = None
        if self.use_regex:
            pattern_parts = []
            for idx, td in enumerate(self.tag_defs):
                # We name the group tag_0, tag_1, etc. to map back to the list index
                pattern_parts.append(f'(?P<tag_{idx}>{re.escape(td.start_marker)})')
            self.start_tag_pattern = re.compile('|'.join(pattern_parts))

    def get_current_active_tag(self) -> TagDefinition:
        """
        Returns the tag currently in effect (the last tag from the bottom up).
        If the stack is empty, returns the default tag.
        """
        return self.stack[-1][0] if self.stack else self.default_tag_type

    def _find_start_tag_regex(self, text: str) -> Tuple[int, int, Optional[TagDefinition]]:
        """
        Regex implementation to find the next start tag.
        """
        if not self.start_tag_pattern:
            return -1, -1, None

        match = self.start_tag_pattern.search(text)
        if match:
            # Identify which tag matched based on group name
            for name, val in match.groupdict().items():
                if val:
                    tag_idx = int(name.split('_')[1])
                    return match.start(), match.end(), self.tag_defs[tag_idx]
        return -1, -1, None

    def _find_start_tag_simple(self, text: str) -> Tuple[int, int, Optional[TagDefinition]]:
        """
        String-only implementation to find the next start tag.
        """
        best_idx = -1
        best_tag = None

        for td in self.tag_defs:
            idx = text.find(td.start_marker)
            if idx != -1:
                # If we found a match earlier than previous best, take it
                if best_idx == -1 or idx < best_idx:
                    best_idx = idx
                    best_tag = td
                    # Optimization: If match is at 0, we can't get better
                    if best_idx == 0:
                        break

        if best_idx != -1 and best_tag:
            return best_idx, best_idx + len(best_tag.start_marker), best_tag

        return -1, -1, None

    def detect_start_tag(self, text: str) -> Tuple[int, int, Optional[TagDefinition]]:
        """
        Public method to check a given string for a start tag.
        Uses the same implementation (Regex or String) as process_stream.

        Returns:
            (start_index, end_index, TagDefinition) or (-1, -1, None)
        """
        if self.use_regex:
            return self._find_start_tag_regex(text)
        else:
            return self._find_start_tag_simple(text)

    def push_tag_context(self, tag_identifier: Union[str, TagDefinition]) -> None:
        """
        Manually forces the parser into a specific tag state.
        Useful when the stream starts with text that is implicitly inside a tag
        (e.g., a "Reasoning" model output that doesn't emit the opening <thought> tag).

        Args:
            tag_identifier: Either the 'tag_type' string (e.g., 'thought')
                            or the TagDefinition object itself.
        """
        target_def = None

        if isinstance(tag_identifier, TagDefinition):
            target_def = tag_identifier
        else:
            # Search by tag_type first
            for td in self.tag_defs:
                if td.tag_type == tag_identifier:
                    target_def = td
                    break

            # Optional: Search by api_tag or start_marker if not found above
            if not target_def:
                for td in self.tag_defs:
                    if td.api_tag == tag_identifier or td.start_marker == tag_identifier:
                        target_def = td
                        break

        if target_def:
            # We push the definition and its expected closing marker onto the stack
            self.stack.append((target_def, target_def.end_marker))
        else:
            raise ValueError(f"Tag definition for '{tag_identifier}' not found in configured definitions.")

    def clear_context(self) -> None:
        """Resets the stack, treating subsequent text as default/plain text."""
        self.stack = []

    def process_stream(self, new_data: str) -> List[Tuple[TagDefinition, str]]:
        self.buffer += new_data
        results = []

        while True:
            current_closer = self.stack[-1][1] if self.stack else None
            current_tag = self.get_current_active_tag()

            # 1. Detect Tags
            start_found_idx, start_found_end, start_found_def = self.detect_start_tag(self.buffer)

            # 2. Detect End Tags
            end_index = -1
            if current_closer:
                end_index = self.buffer.find(current_closer)

            # --- Logic Branching ---

            # Case A: Closing Tag Found
            if end_index != -1 and (start_found_idx == -1 or end_index < start_found_idx):
                content = self.buffer[:end_index]
                if content:
                    results.append((current_tag, content))

                # Remove content + closer from buffer
                self.buffer = self.buffer[end_index + len(current_closer):]
                self.stack.pop()
                continue

            # Case B: New Start Tag Found
            elif start_found_idx != -1:
                pre_content = self.buffer[:start_found_idx]
                if pre_content:
                    results.append((current_tag, pre_content))

                if start_found_def:
                    self.stack.append((start_found_def, start_found_def.end_marker))

                self.buffer = self.buffer[start_found_end:]
                continue

            # Case C: AGGRESSIVE STREAMING (Corrected)
            # If no tags are detected in the current buffer, we flush the text
            # to the client immediately.
            # HOWEVER, we must keep the "tail" of the buffer just in case
            # the last few characters are the beginning of a tag (e.g., "<too")
            # that will be completed in the next stream chunk.
            else:
                margin = self.max_marker_length

                # Only flush if we have more text than the safety margin
                if len(self.buffer) > margin:
                    # We flush everything except the last 'margin' characters
                    chunk = self.buffer[:-margin]
                    results.append((current_tag, chunk))

                    # Keep the tail in the buffer for the next iteration
                    self.buffer = self.buffer[-margin:]

                # Break the loop to wait for more network data
                break

        return results

    def flush(self) -> List[Tuple[str, str]]:
        """
        Forces the remaining buffer to be returned as a result.
        """
        results = []
        if self.buffer:
            current_tag = self.get_current_active_tag()
            results.append((current_tag, self.buffer))
            self.buffer = ""
        return results


    def parse_full_text(
        self,
        text: str,
        initial_tag: Union[str, TagDefinition] = None
    ) -> List[Tuple[TagDefinition, str]]:
        """
        Parses a complete string and merges consecutive segments with the same tag_type.

        Args:
            text: The content to parse.
            initial_tag: Optional tag to force the parser into a specific state
                         (e.g., 'thought') before processing starts.
        """
        # 1. Reset internal state to ensure a clean parse
        self.buffer = ""
        self.stack = []

        # 2. Apply initial context if provided
        if initial_tag:
            self.push_tag_context(initial_tag)

        # 3. Process the text as one giant stream chunk.
        # This yields a list of (TagDefinition, text fragment)
        raw_results = self.process_stream(text)

        # 4. Flush the remaining safety margin.
        raw_results.extend(self.flush())

        # 4.1 Remove empty string/ newlines
        raw_results = [_result for _result in raw_results if _result[1].strip()]

        # 5. Merge consecutive items with the same tag_type
        merged_results: List[Tuple[TagDefinition, str]] = []

        for tag, content in raw_results:
            # If we have existing results and the last one matches the current tag_type
            if merged_results and merged_results[-1][0].tag_type == tag.tag_type:
                prev_tag, prev_content = merged_results.pop()
                # Combine the previous content with the new content
                merged_results.append((prev_tag, prev_content + content))
            else:
                # Otherwise, start a new entry
                merged_results.append((tag, content))

        return merged_results