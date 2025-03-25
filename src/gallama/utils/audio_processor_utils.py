import librosa
import numpy as np
import io
import soundfile as sf
from gallama.logger.logger import logger
import time
import asyncio
from ..data_classes import TTSEvent


class AudioProcessor:
    def __init__(
        self,
        input_sample_rate: int = 32000,
        output_sample_rate: int = 24000,
        min_chunk_size: int = 1024,
        optimal_chunk_size: int = 32768,
        format='wav',
        subtype='PCM_16'
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.format = format
        self.subtype = subtype
        self.buffer = bytearray()
        self.min_chunk_size = min_chunk_size
        self.optimal_chunk_size = optimal_chunk_size
        self.total_processed = 0
        self.chunks_processed = 0
        self.process_start_time = None
        self.bytes_per_sample = 2  # For PCM16

    def process_chunk(self, audio_data, sample_rate=None):
        """Process audio chunk with dynamic chunk sizing and buffer management"""
        MAX_BUFFER_SIZE = 1024 * 1024  # 1MB maximum buffer size

        if self.process_start_time is None:
            self.process_start_time = time.time()

        if sample_rate is not None:
            self.input_sample_rate = sample_rate

        # Convert numpy array to bytes if needed
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.tobytes()

        original_buffer_size = len(self.buffer)
        self.buffer.extend(audio_data)
        new_buffer_size = len(self.buffer)

        logger.info(
            f"Buffer state - Previous: {original_buffer_size} bytes, Added: {len(audio_data)} bytes, New: {new_buffer_size} bytes")

        # Emergency buffer size control
        if new_buffer_size > MAX_BUFFER_SIZE:
            logger.warning(f"Buffer exceeded {MAX_BUFFER_SIZE} bytes, processing aggressively")
            results = []
            # Process in larger chunks until buffer is under control
            while len(self.buffer) > MAX_BUFFER_SIZE // 2:
                chunk_size = min(len(self.buffer) // 2, self.optimal_chunk_size * 4)
                chunk = self.buffer[:chunk_size * self.bytes_per_sample]
                result = self._process_audio_chunk(chunk)
                if result:
                    results.append(result)
                self.buffer = self.buffer[chunk_size * self.bytes_per_sample:]
            return b''.join(results) if results else None

        # Check if we have enough data
        samples_available = len(self.buffer) // self.bytes_per_sample
        logger.info(f"Samples available: {samples_available}, Minimum needed: {self.min_chunk_size}")

        if samples_available < self.min_chunk_size:
            logger.info("Not enough samples, buffering...")
            return None

        # Dynamic chunk sizing based on buffer fullness
        target_chunk_size = self.optimal_chunk_size
        if samples_available > self.optimal_chunk_size * 2:
            # Gradually increase chunk size as buffer grows
            buffer_multiple = samples_available / self.optimal_chunk_size
            target_chunk_size = min(
                int(self.optimal_chunk_size * min(buffer_multiple / 2, 4)),
                samples_available // 2
            )
            logger.info(f"Buffer getting full, processing larger chunk: {target_chunk_size} samples")

        # Extract chunk to process
        bytes_to_process = target_chunk_size * self.bytes_per_sample
        chunk = self.buffer[:bytes_to_process]
        self.buffer = self.buffer[bytes_to_process:]

        self.chunks_processed += 1
        self.total_processed += target_chunk_size

        processing_time = time.time() - self.process_start_time
        logger.info(
            f"Processed {self.chunks_processed} chunks, {self.total_processed} samples in {processing_time:.2f}s")
        logger.info(f"Average throughput: {self.total_processed / processing_time:.2f} samples/second")

        return self._process_audio_chunk(chunk)

    def _process_audio_chunk(self, audio_bytes):
        """Process a single chunk of audio data with optimized resampling"""
        BATCH_SIZE = 32768  # 32KB batch size for large chunks
        process_start = time.time()

        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Only resample if rates differ
            if self.input_sample_rate != self.output_sample_rate:
                # Convert to float32 for librosa
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Use faster resampling algorithm for real-time processing
                audio_resampled = librosa.resample(
                    audio_float,
                    orig_sr=self.input_sample_rate,
                    target_sr=self.output_sample_rate,
                    res_type='kaiser_fast'  # Faster resampling algorithm
                )

                # Convert back to int16
                audio_array = (audio_resampled * 32768).astype(np.int16)

            # For large chunks, process all the data at once instead of batching
            buffer = io.BytesIO()
            sf.write(
                buffer,
                audio_array,
                self.output_sample_rate,
                format=self.format,
                subtype=self.subtype
            )
            buffer.seek(0)

            process_duration = time.time() - process_start
            if process_duration > 0.1:  # Log warning if processing takes too long
                logger.warning(f"Chunk processing took {process_duration * 1000:.2f}ms")

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error in audio chunk processing: {str(e)}")
            return None


    def flush(self):
        """Process any remaining audio in the buffer"""
        if len(self.buffer) >= self.min_chunk_size * self.bytes_per_sample:
            logger.info(f"Flushing remaining {len(self.buffer)} bytes from buffer")
            result = self._process_audio_chunk(self.buffer)
            self.buffer = bytearray()
            return result
        return None