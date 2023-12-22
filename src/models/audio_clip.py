class AC:
    """
    Represents audio that is ready to be processed
    """

    def __init__(self, array, samplerate):
        self.array = array
        self.samplerate = samplerate
