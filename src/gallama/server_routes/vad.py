class VADProcessor:
    def __init__(self):
        # Load Silero VAD model using the package's loader
        self.model = load_silero_vad()
        self.model.eval()

        # Initialize VAD iterator
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=0.5,  # sensitivity threshold, higher = more sensitive
            sampling_rate=16000,  # expected sample rate
            min_speech_duration_ms=250,  # minimum speech chunk duration
            max_speech_duration_s=float('inf'),  # maximum speech chunk duration
            min_silence_duration_ms=500  # minimum silence duration to split speech
        )

        # Buffer for accumulating audio
        self.audio_buffer = []
        self.silence_duration = 0
        self.SILENCE_THRESHOLD = 1000  # ms of silence to consider end of speech

    def process_audio_chunk(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """
        Process an audio chunk and determine if it contains speech and if it's end of speech
        Returns: (contains_speech, is_end_of_speech)
        """
        # Convert bytes to numpy array (assuming 16-bit PCM audio)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

        # Convert to float32 and normalize
        audio_float = audio_np.astype(np.float32) / 32768.0

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_float)

        # Process with VAD
        speech_prob = self.vad_iterator(audio_tensor, return_seconds=True)

        # Update silence duration
        if speech_prob < 0.5:  # No speech detected
            self.silence_duration += len(audio_np) / 16000 * 1000  # Convert to ms
        else:
            self.silence_duration = 0

        # Check if we've hit silence threshold
        is_end_of_speech = self.silence_duration >= self.SILENCE_THRESHOLD

        if is_end_of_speech:
            self.silence_duration = 0  # Reset silence counter

        return speech_prob >= 0.5, is_end_of_speech