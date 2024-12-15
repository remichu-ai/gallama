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


    def insert(self, new: TimeStampedWord, offset: float):  # TODO
        """
        Adjusts the timestamps of the new hypotheses by offset.
        Filters out any hypotheses that are older than last_commited_time - 0.1.
        Removes overlapping words between commited_in_buffer and new (using n-gram matching).

        How the algorithm works:
        compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        the new tail is added to self.new
        """

        new = [(a + offset, b + offset, t) for a, b, t in new]

        # Filter based on timestamp
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # N-gram matching for overlap removal
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
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
