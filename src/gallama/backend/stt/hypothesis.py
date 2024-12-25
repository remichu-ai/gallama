from typing import List, Tuple
from gallama.logger.logger import logger
from ...data_classes import TimeStampedWord


class HypothesisBuffer:

    def __init__(self):
        """
        This class manages hypotheses (potential recognized words) from the ASR system and determines which words can be committed as finalized transcriptions.
        """
        self.commited_in_buffer = []  # Stores finalized (committed) transcriptions.
        self.buffer = []  # Stores new hypotheses inserted but not yet processed.
        self.new: List[Tuple[float, float, str]] = []  # Stores new hypotheses inserted but not yet processed.

        self.last_commited_time = 0  # Timestamp of the last committed word.
        self.last_commited_word = None  # Last committed word.
        self.is_final = False  # New flag for final processing


        # Parameters for detecting repetition patterns
        self.min_word_gap = 0.1  # Minimum expected gap between distinct words
        self.max_merge_window = 2.0  # Maximum window to consider for merging

    def detect_repetition_pattern(self, words):
        """
        Analyze sequence of words to detect abnormal repetition patterns
        Returns list of indices that are likely part of a repetition pattern
        """
        if len(words) < 2:
            return []

        repetition_indices = []
        for i in range(1, len(words)):
            prev_time = words[i - 1][1]  # End time of previous word
            curr_time = words[i][0]  # Start time of current word
            time_gap = curr_time - prev_time

            # Check for suspiciously small gaps between words
            if time_gap < self.min_word_gap:
                # Look for repetition of single characters or very short segments
                if len(words[i][2]) <= 2 and words[i][2] == words[i - 1][2]:
                    repetition_indices.extend([i - 1, i])

        return list(set(repetition_indices))

    def merge_adjacent_segments(self, words):
        """
        Merge adjacent word segments that might be part of the same word
        """
        if len(words) < 2:
            return words

        merged = []
        i = 0
        while i < len(words):
            if i == len(words) - 1:
                merged.append(words[i])
                break

            curr_word = words[i]
            next_word = words[i + 1]

            # Check if these segments should be merged
            time_gap = next_word[0] - curr_word[1]
            if time_gap < self.min_word_gap and len(curr_word[2]) <= 2 and len(next_word[2]) <= 2:
                # Merge the segments
                merged_text = curr_word[2] + next_word[2]
                merged_segment = (curr_word[0], next_word[1], merged_text)
                merged.append(merged_segment)
                i += 2
            else:
                merged.append(curr_word)
                i += 1

        return merged

    def insert(self, new: TimeStampedWord, offset: float):  # TODO
        """
        Adjusts the timestamps of the new hypotheses by offset.
        Filters out any hypotheses that are older than last_commited_time - 0.1.
        Removes overlapping words between commited_in_buffer and new (using n-gram matching).

        How the algorithm works:
        compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        the new tail is added to self.new
        """

        # Apply offset to timestamps
        new = [(a + offset, b + offset, t) for a, b, t in new]

        # Filter based on timestamp
        filtered_new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.5]

        if filtered_new:
            # Detect repetition patterns
            rep_indices = self.detect_repetition_pattern(filtered_new)

            # Remove detected repetitions
            filtered_new = [word for i, word in enumerate(filtered_new)
                            if i not in rep_indices]

            # Merge adjacent segments that might be part of the same word
            filtered_new = self.merge_adjacent_segments(filtered_new)

            self.new = filtered_new

            # Perform n-gram matching with existing buffer
            if self.new and abs(self.new[0][0] - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join([self.commited_in_buffer[-j][2]
                                      for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2]
                                        for j in range(1, i + 1))
                        if c == tail:
                            for j in range(i):
                                self.new.pop(0)
                            break

    def flush(self, force_commit=False):
        """
        Commits words to the final transcription.
        Args:
            force_commit: If True, commits remaining words without waiting for confirmation
        """
        commit = []

        if force_commit:
            # Commit all remaining words in buffer and new
            remaining_words = []
            if self.buffer:
                remaining_words.extend(self.buffer)
            if self.new:
                remaining_words.extend(self.new)

            # Sort by start time
            remaining_words.sort(key=lambda x: x[0])

            # Remove duplicates while preserving order
            seen_words = {}  # Using dict to preserve order
            for na, nb, nt in remaining_words:
                # If we see the same word again, keep the one with the later timestamp
                if nt in seen_words:
                    prev_na, prev_nb, _ = seen_words[nt]
                    if nb > prev_nb:
                        seen_words[nt] = (na, nb, nt)
                else:
                    seen_words[nt] = (na, nb, nt)

            # Convert back to list
            commit = list(seen_words.values())

            # Update last committed info
            if commit:
                self.last_commited_word = commit[-1][2]
                self.last_commited_time = commit[-1][1]

            self.buffer = []
            self.new = []

        else:
            # Normal matching-based commitment
            while self.new and self.buffer:
                na, nb, nt = self.new[0]
                if nt == self.buffer[0][2]:
                    commit.append((na, nb, nt))
                    self.last_commited_word = nt
                    self.last_commited_time = nb
                    self.buffer.pop(0)
                    self.new.pop(0)
                else:
                    break

            self.buffer = self.new
            self.new = []

        self.commited_in_buffer.extend(commit)
        return commit

    def set_final(self):
        """Mark the buffer as final, triggering complete word commitment"""
        self.is_final = True
        return self.flush(force_commit=True)

    def pop_committed(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Returns all remaining words in the buffer"""
        return self.buffer
