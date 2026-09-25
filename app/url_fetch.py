"""Fetch a test video from a URL (YouTube, Shorts, etc.) via yt-dlp.

Testing convenience only — the real product flow is a user uploading their
own clip. Picks an H.264 MP4 stream so OpenCV can read it and no ffmpeg
merge step is needed; video-only streams are fine since the pipeline never
uses audio.
"""
from __future__ import annotations

from pathlib import Path

# Single-file H.264 mp4 first, then video-only H.264 mp4 (YouTube Shorts often
# only offer separate streams), capped at 1080p — the pipeline downsizes to 720p anyway.
FORMAT = (
    "best[ext=mp4][vcodec^=avc1][height<=1080]"
    "/bv*[ext=mp4][vcodec^=avc1][height<=1080]"
    "/best[ext=mp4][height<=1080]"
)
MAX_FILESIZE_BYTES = 200 * 1024 * 1024


def download_video_from_url(url: str, dest_dir: Path) -> Path:
    """Downloads the video at `url` into `dest_dir` and returns its path.
    Raises RuntimeError with a user-readable message on any failure."""
    url = (url or "").strip().strip("<>")
    if not url.startswith(("http://", "https://")):
        raise RuntimeError("Enter a full video URL starting with http:// or https://.")

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp isn't installed. Run `pip install -r requirements.txt`.") from e

    dest_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": FORMAT,
        "outtmpl": str(dest_dir / "url_%(id)s.%(ext)s"),
        "noplaylist": True,
        "max_filesize": MAX_FILESIZE_BYTES,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Couldn't download that video: {e}") from e

    if not path.exists():
        raise RuntimeError("Download finished but no MP4 file was produced (it may exceed the size limit).")
    return path
