"""Video sources: recorded files, USB webcams and RTSP cameras."""

from sopvision.sources.base import Frame, SourceUnavailableError, VideoSource

__all__ = ["Frame", "SourceUnavailableError", "VideoSource"]
