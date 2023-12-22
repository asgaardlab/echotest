class AudioSubtitleMatch:
    def __init__(self, audio_token, subtitle_token, score):
        self.audio_token = audio_token
        self.subtitle_token = subtitle_token if score else None
        self.score = score

    def __eq__(self, other):
        if not isinstance(other, AudioSubtitleMatch):
            return False
        return (
            self.audio_token == other.audio_token
            and self.subtitle_token == other.subtitle_token
            and self.score == other.score
        )

    def __str__(self):
        subtitle_str = f"Subtitle: {self.subtitle_token}" if self.subtitle_token else "No Subtitle"
        return f"Audio Token: {self.audio_token}\n{subtitle_str}\nScore: {self.score}"