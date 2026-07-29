from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def normalise_source(source: str | int) -> str | int:
    """
    Convert a numeric string such as '0' into an integer camera index.

    Other strings are retained as file paths or RTSP URLs.
    """
    if isinstance(source, int):
        return source

    source_text = str(source).strip()

    if source_text.isdigit():
        return int(source_text)

    return source_text


class VideoSource:
    """
    OpenCV video-source abstraction supporting:

    - Local video files.
    - USB cameras.
    - RTSP network cameras.
    - Local-video looping.
    - Basic camera/RTSP reconnection.
    """

    def __init__(
        self,
        source: str | int,
        loop_local_file: bool = True,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self.source = normalise_source(source)
        self.loop_local_file = loop_local_file
        self.reconnect_delay_s = reconnect_delay_s

        self._capture: cv2.VideoCapture | None = None
        self._last_open_attempt = 0.0

    @property
    def is_rtsp(self) -> bool:
        """Return True when the source is an RTSP URL."""
        return isinstance(self.source, str) and self.source.lower().startswith(
            ("rtsp://", "rtsps://")
        )

    @property
    def is_local_file(self) -> bool:
        """Return True when the source appears to be a local video file."""
        if not isinstance(self.source, str):
            return False

        if self.is_rtsp:
            return False

        return Path(self.source).suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".m4v",
        }

    @property
    def is_open(self) -> bool:
        """Return True if OpenCV currently has an open capture."""
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> bool:
        """
        Open the configured source.

        Returns:
            True if the source opened successfully.
        """
        self.release()
        self._last_open_attempt = time.monotonic()

        self._capture = cv2.VideoCapture(self.source)

        # Some backends ignore this property, which is acceptable.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            self.release()
            return False

        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame.

        For a completed local video, the video is restarted when
        loop_local_file=True.

        For an unavailable RTSP or USB camera, the source is released and
        reopened after reconnect_delay_s.
        """
        if not self.is_open:
            elapsed = time.monotonic() - self._last_open_attempt

            if elapsed < self.reconnect_delay_s:
                return False, None

            if not self.open():
                return False, None

        assert self._capture is not None

        success, frame = self._capture.read()

        if success and frame is not None:
            return True, frame

        if self.is_local_file and self.loop_local_file:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._capture.read()

            if success and frame is not None:
                return True, frame

        # RTSP/USB failure or unrecoverable file failure.
        self.release()
        return False, None

    def change_source(self, source: str | int) -> None:
        """Release the current source and select a new one."""
        self.release()
        self.source = normalise_source(source)
        self._last_open_attempt = 0.0

    def release(self) -> None:
        """Release the OpenCV capture safely."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
