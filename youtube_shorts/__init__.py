"""
YouTube Shorts Automation Suite

A collection of tools for automating the creation, optimization, and management of YouTube Shorts.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main modules
# setup_workspace and downloader were removed from the package; the two
# imports that named them are gone with them.
from . import performance_tracker
from . import uploader
from . import youtube_limits
