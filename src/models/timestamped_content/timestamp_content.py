import functools
from typing import List, Tuple, Dict, Union

TIMESTAMP = "timestamp"


class TimestampedContent:
    def __init__(self, timestamp: Union[Tuple[float, float], 'Timestamp'], content: any):
        if isinstance(timestamp, Timestamp):
            self.timestamp = timestamp
        else:
            self.timestamp = Timestamp(start=timestamp[0], end=timestamp[1] if timestamp[1] else timestamp[0] + 1)
        self.content = content

    def __str__(self):
        """
        Returns a string representation of the TimestampedContent object.
        """
        return f"Timestamp: {self.timestamp.start:.2f}-{self.timestamp.end:.2f}, Content: {self.content}"

    def merge(self, other: 'TimestampedContent') -> 'TimestampedContent':
        """
        Merges two timestamps together, timestamps must be compatible to merge:
        to be compatible one timestamp must finish at the same time the next completes
        :param other: timestamp to combine
        :return: merged timestamp
        """
        if self.timestamp.end > other.timestamp.start:
            raise ValueError("Timestamps must be compatible for merge")
        new_start = min(self.timestamp.start, other.timestamp.start)
        new_end = max(self.timestamp.end, other.timestamp.end)
        new_content = self.content
        if not self.content.endswith(other.content) and other.content.strip():
            new_content += ' ' + other.content
        return TimestampedContent(timestamp=(new_start, new_end), content=new_content)


class Timestamp:
    def __init__(self, start: float, end: float):
        if start < end:
            self.start = start
            self.end = end
        else:
            raise ValueError("Start of timestamp must be before the end")

    def overlaps(self, other: 'Timestamp') -> bool:
        """
        Two timestamps overlap in the case that the two are active at the same time
        :param other: another timestamp to compare to
        """
        return self.end > other.start and self.start < other.end

    def get_overlap(self, other: 'Timestamp') -> 'Timestamp':
        """
        Returns the overlap between two timestamped contents
        :param other: another timestamp to compare to
        """
        return Timestamp(start=max(self.start, other.start), end=min(self.end, other.end))


def from_dict(timestamp_dict: Dict[str, any], content_key: str) -> List[TimestampedContent]:
    """
    Creates an array of timestamped content from a dictionary
    :param timestamp_dict: timestamped dictionary from which to create `TimestampContent` list
    :param content_key: represents the dictionary key which contains the timestamped content
    """
    return list(map(lambda chunk: TimestampedContent(timestamp=chunk[TIMESTAMP],
                                                     content=chunk[content_key]),
                    timestamp_dict))


def merge_sequential_timestamps(timestamped_contents: List[TimestampedContent]) -> TimestampedContent:
    """
    Combines a list of sequential timestamps into a single timestamp by merging them together into a single entry
    :param timestamped_contents: to merge
    """
    return functools.reduce(lambda x, y: x.merge(y), timestamped_contents)


def get_overlapping_segments(segment_a: List[TimestampedContent], segment_b: List[TimestampedContent]):
    """
    Returns a list of tuples containing overlapping content between two lists of TimestampedContent objects.

    :param segment_a: A list of TimestampedContent objects representing the first segment.
    :param segment_b: A list of TimestampedContent objects representing the second segment.
    :return: A list of tuples containing overlapping content between segment_a and segment_b.
             Each tuple contains the content from the overlapping segments of segment_a and segment_b.
    """
    overlapping_segments: List[TimestampedContent] = []
    for segA in segment_a:
        for segB in segment_b:
            if segA.timestamp.overlaps(segB.timestamp):
                overlapping_segments.append(
                    TimestampedContent(segA.timestamp.get_overlap(segB.timestamp), (segA.content, segB.content)))
    return overlapping_segments
