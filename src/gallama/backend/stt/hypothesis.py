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

        # Filters out any hypotheses that are older than last_commited_time - 0.1.
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

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

    def pop_committed(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer

