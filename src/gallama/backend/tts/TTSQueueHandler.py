from typing import Any, AsyncIterator
import asyncio
from gallama.logger.logger import logger


class TTSQueueHandler:
    """Handles text segment processing and audio streaming for TTS with proper synchronization"""

    def __init__(self, tts_engine: "TTSBase"):
        self.tts_engine = tts_engine
        self.text_queue = asyncio.Queue()  # Queue for text segments
        self.audio_queue = asyncio.Queue()  # Queue for audio chunks
        self._processing = False
        self._current_task = None

    async def start(self):
        """Start the processing loop"""
        self._processing = True
        self._current_task = asyncio.create_task(self._process_segments())

    async def stop(self):
        """Stop the processing loop"""
        self._processing = False
        if self._current_task:
            await self._current_task
            self._current_task = None
        # Clear queues
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def add_segment(self, text: str):
        """Add a text segment to be processed"""
        await self.text_queue.put(text)

    async def get_audio(self):
        """Get the next audio chunk"""
        return await self.audio_queue.get()

    async def _process_segments(self):
        """Main processing loop that handles text segments one at a time"""
        while self._processing:
            try:
                # Get the next text segment
                text = await self.text_queue.get()
                if text is None:  # Check for stop signal
                    break

                # Create a temporary queue for this segment's audio chunks
                segment_queue = asyncio.Queue()

                # Process the current segment
                try:
                    await self.tts_engine.text_to_speech(
                        text=text,
                        queue=segment_queue,
                        stream=True,
                        language="auto"
                    )

                    # Forward all audio chunks to the main audio queue
                    while True:
                        chunk = await segment_queue.get()
                        if chunk is None:  # End of segment
                            break
                        if isinstance(chunk, Exception):
                            raise chunk
                        await self.audio_queue.put(chunk)

                except Exception as e:
                    logger.error(f"Error processing segment: {str(e)}")
                    await self.audio_queue.put(Exception(f"TTS error: {str(e)}"))
                    break

                # Mark this segment as done
                self.text_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in segment processing: {str(e)}")
                await self.audio_queue.put(Exception(f"Unexpected error: {str(e)}"))
                break

        # Signal end of processing
        await self.audio_queue.put(None)