"""
YouTube Uploader with Page Object Model

This module provides functions for uploading videos to YouTube using the Page Object Model pattern.
It replaces the monolithic upload_video function with smaller, focused functions.

Copyright (c) 2023-2025 Shahid Ali
License: MIT License
GitHub: https://github.com/Mrshahidali420/youtube-shorts-automation
Version: 1.0.0
"""

import os
import time
import json
import logging
import subprocess
import platform
import signal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

# --- NEW: Import constants from the new location ---
try:
    # Assuming uploader_pom.py is in youtube_shorts/
    # and constants.py is in youtube_shorts/utils/
    from .utils import constants
    CONSTANTS_IMPORTED = True
except ImportError:
    try:
        from utils import constants # Fallback if utils is directly in PYTHONPATH
        CONSTANTS_IMPORTED = True
    except ImportError:
        try:
            from youtube_shorts.utils import constants
            CONSTANTS_IMPORTED = True
        except ImportError:
            CONSTANTS_IMPORTED = False
            print("CRITICAL: uploader_pom.py could not import constants.py. Path issues.")
            class MinimalConstants: # Fallback
                DEBUG_RECORDINGS_DIR = "output/debug_recordings"
                LOGS_DIR = "logs"
                UPLOADER_POM_LOG_FILE = "logs/uploader_pom.log"
                CONFIG_FILE_PATH = "config/config.txt"
                DEFAULT_YOUTUBE_DESCRIPTION_LIMIT = 4900
                DEFAULT_YOUTUBE_TAG_LIMIT = 100
                DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT = 450
                DEFAULT_YOUTUBE_MAX_TAGS_COUNT = 40
                GECKODRIVER_PATH = None
            constants = MinimalConstants()
            print("WARNING: uploader_pom.py using minimal fallback constants.")
# --- End NEW Import ---

# Import selenium components
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.common.exceptions import WebDriverException
from webdriver_manager.firefox import GeckoDriverManager

# Import page objects
try:
    from .page_objects import ( # if page_objects is in the same package level
        StudioHomePage,
        UploadPage,
        DetailsPage,
        VisibilityPage,
        ConfirmationPage
    )
except ImportError:
    try:
        from page_objects import ( # Fallback if running from a different context
            StudioHomePage,
            UploadPage,
            DetailsPage,
            VisibilityPage,
            ConfirmationPage
        )
    except ImportError:
        print("CRITICAL: Could not import page_objects. Path issues.")
        raise

# Import YouTube limits
try:
    from .youtube_limits import ( # if youtube_limits.py is in the same package level
        validate_description,
        validate_tags,
        DEFAULT_YOUTUBE_DESCRIPTION_LIMIT,
        DEFAULT_YOUTUBE_TAG_LIMIT,
        DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT,
        DEFAULT_YOUTUBE_MAX_TAGS_COUNT
    )
except ImportError:
    try:
        from . import youtube_limits # Fallback when it sits next to this file
        validate_description = youtube_limits.validate_description
        validate_tags = youtube_limits.validate_tags
        DEFAULT_YOUTUBE_DESCRIPTION_LIMIT = youtube_limits.DEFAULT_YOUTUBE_DESCRIPTION_LIMIT
        DEFAULT_YOUTUBE_TAG_LIMIT = youtube_limits.DEFAULT_YOUTUBE_TAG_LIMIT
        DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT = youtube_limits.DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT
        DEFAULT_YOUTUBE_MAX_TAGS_COUNT = youtube_limits.DEFAULT_YOUTUBE_MAX_TAGS_COUNT
    except ImportError:
        # Fallback constants
        print("WARNING: youtube_limits.py not found. Using fallback validation and constants.")
        DEFAULT_YOUTUBE_DESCRIPTION_LIMIT = constants.DEFAULT_YOUTUBE_DESCRIPTION_LIMIT if CONSTANTS_IMPORTED else 4900
        DEFAULT_YOUTUBE_TAG_LIMIT = constants.DEFAULT_YOUTUBE_TAG_LIMIT if CONSTANTS_IMPORTED else 100
        DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT = constants.DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT if CONSTANTS_IMPORTED else 450
        DEFAULT_YOUTUBE_MAX_TAGS_COUNT = constants.DEFAULT_YOUTUBE_MAX_TAGS_COUNT if CONSTANTS_IMPORTED else 40

    # Dummy validation functions
    def validate_description(desc: str, limit: int = DEFAULT_YOUTUBE_DESCRIPTION_LIMIT) -> Tuple[str, List[str]]:
        warnings = []
        validated_desc = desc
        if len(desc) > limit:
            warnings.append(f"Description truncated from {len(desc)} to {limit} characters.")
            validated_desc = desc[:limit]
        return validated_desc, warnings

    def validate_tags(tags: List[str], tag_char_limit: int = DEFAULT_YOUTUBE_TAG_LIMIT,
                     total_char_limit: int = DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT,
                     max_count_limit: int = DEFAULT_YOUTUBE_MAX_TAGS_COUNT) -> Tuple[List[str], List[str]]:
        warnings = []
        validated_tags = []
        current_total_chars = 0
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                warnings.append(f"Skipped invalid/empty tag: '{tag}'")
                continue
            cleaned_tag = tag.strip()
            if len(validated_tags) >= max_count_limit:
                warnings.append(f"Tag count limit ({max_count_limit}) reached. Skipped tag: '{cleaned_tag[:30]}...'")
                continue
            if len(cleaned_tag) > tag_char_limit:
                warnings.append(f"Tag truncated from {len(cleaned_tag)} to {tag_char_limit} chars: '{cleaned_tag[:30]}...'")
                cleaned_tag = cleaned_tag[:tag_char_limit]
            potential_len = current_total_chars + len(cleaned_tag) + (1 if validated_tags else 0)
            if potential_len > total_char_limit:
                warnings.append(f"Total tag character limit ({total_char_limit}) reached. Skipped tag: '{cleaned_tag[:30]}...'")
                continue
            validated_tags.append(cleaned_tag)
            current_total_chars += len(cleaned_tag) + (1 if len(validated_tags) > 1 else 0)
        return validated_tags, warnings

# Constants
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEBUG_RECORDING_FOLDER_PATH = constants.DEBUG_RECORDINGS_DIR if CONSTANTS_IMPORTED else "output/debug_recordings"

# Global variables for recording
_current_recording_process: Optional[subprocess.Popen] = None
_current_recording_filename: Optional[str] = None

# Configure logging
logger = logging.getLogger("uploader_pom") # Use specific name
if not logger.handlers:
    # Ensure logs directory exists using constants
    log_file_dir = constants.LOGS_DIR if CONSTANTS_IMPORTED else "logs"
    os.makedirs(log_file_dir, exist_ok=True)
    uploader_pom_log_path = constants.UPLOADER_POM_LOG_FILE if CONSTANTS_IMPORTED else os.path.join(log_file_dir, "uploader_pom.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(uploader_pom_log_path),
            logging.StreamHandler()
        ]
    )

def setup_browser(profile_path: Optional[str] = None) -> Optional[webdriver.Firefox]:
    """
    Set up and return a Firefox browser instance.

    Args:
        profile_path: Path to Firefox profile (optional)

    Returns:
        Firefox WebDriver instance or None if setup failed
    """
    logger.info("Setting up Firefox browser...")

    try:
        # Configure Firefox options
        options = FirefoxOptions()

        # Add profile if specified
        if profile_path and os.path.exists(profile_path):
            logger.info(f"Using Firefox profile: {profile_path}")
            options.add_argument(f"-profile")
            options.profile = profile_path # For newer Selenium versions
        else:
            if profile_path:
                logger.warning(f"Firefox profile path '{profile_path}' not found. Using default.")

        # Set up Firefox driver
        # Try to get GeckoDriver path from constants, then fall back to WebDriverManager
        gecko_driver_path_const = getattr(constants, 'GECKODRIVER_PATH', None) if CONSTANTS_IMPORTED else None

        if gecko_driver_path_const and os.path.exists(gecko_driver_path_const):
            logger.info(f"Using GeckoDriver from constants: {gecko_driver_path_const}")
            service = FirefoxService(executable_path=gecko_driver_path_const)
        else:
            if gecko_driver_path_const: # Path was defined but not found
                logger.warning(f"GeckoDriver not found at constants path '{gecko_driver_path_const}'. Using WebDriverManager.")
            else: # Path not in constants
                logger.info("GeckoDriver path not in constants. Using WebDriverManager.")
            service = FirefoxService(GeckoDriverManager().install())

        driver = webdriver.Firefox(service=service, options=options)

        # Maximize window
        driver.maximize_window()

        logger.info("Firefox browser set up successfully")
        return driver
    except Exception as e:
        logger.error(f"Failed to set up Firefox browser: {e}")
        return None

def start_recording(video_index: str, ffmpeg_cmd_path: str, driver: webdriver.Firefox) -> Optional[Tuple[subprocess.Popen, str]]:
    """
    Start screen recording using FFmpeg.

    Args:
        video_index: Video index for filename
        ffmpeg_cmd_path: Path to FFmpeg executable
        driver: WebDriver instance

    Returns:
        Tuple of (recording process, output filename) or None if recording failed
    """
    global _current_recording_process, _current_recording_filename

    if not driver:
        logger.error("Cannot start recording: WebDriver is not valid")
        return None

    # Create debug recordings folder if it doesn't exist
    if not os.path.exists(DEBUG_RECORDING_FOLDER_PATH):
        os.makedirs(DEBUG_RECORDING_FOLDER_PATH)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(DEBUG_RECORDING_FOLDER_PATH, f"upload_debug_{video_index}_{timestamp}.mp4")

    try:
        # Get window title for targeting
        window_title = driver.title

        # Prepare FFmpeg command based on OS
        if platform.system() == "Windows":
            # Windows: Use GDI grabber
            cmd = [
                ffmpeg_cmd_path,
                "-f", "gdigrab",
                "-framerate", "10",
                "-i", f"title={window_title}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                output_filename
            ]
        else:
            # Linux/Mac: Use X11 grabber
            cmd = [
                ffmpeg_cmd_path,
                "-f", "x11grab",
                "-framerate", "10",
                "-i", ":0.0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                output_filename
            ]

        logger.info(f"Starting FFmpeg recording: {' '.join(cmd)}")

        # Start FFmpeg process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )

        # Check if process started successfully
        if process.poll() is not None:
            stderr_output = process.stderr.read().decode('utf-8', errors='replace') if process.stderr else ""
            logger.error(f"FFmpeg failed to start (return code: {process.returncode})")
            if stderr_output:
                logger.error(f"FFmpeg error output: {stderr_output}")
            return None

        logger.info(f"Recording started: {output_filename}")
        _current_recording_process = process
        _current_recording_filename = output_filename
        return process, output_filename

    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return None

def stop_recording(process: subprocess.Popen, filename: str, keep_recording: bool = True) -> None:
    """
    Stop the FFmpeg recording process.

    Args:
        process: Recording process
        filename: Output filename
        keep_recording: Whether to keep the recording file
    """
    global _current_recording_process, _current_recording_filename

    if not process:
        return

    logger.info(f"Stopping recording: {filename}")

    try:
        # Send termination signal based on OS
        if platform.system() == "Windows":
            # Windows: Use taskkill
            process.terminate()
        else:
            # Linux/Mac: Send SIGINT (Ctrl+C)
            process.send_signal(signal.SIGINT)

        # Wait for process to terminate
        process.wait(timeout=5)

        # Reset global variables
        if _current_recording_process == process:
            _current_recording_process = None
            _current_recording_filename = None

        # Delete recording if not needed
        if not keep_recording and os.path.exists(filename):
            logger.info(f"Deleting recording: {filename}")
            os.remove(filename)

    except Exception as e:
        logger.error(f"Error stopping recording: {e}")

        # Force kill if termination failed
        try:
            process.kill()
        except:
            pass

def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean up metadata before upload.

    Args:
        metadata: Original metadata dictionary

    Returns:
        Validated metadata dictionary
    """
    validated = metadata.copy()

    # Get title
    title = validated.get("title", "") or validated.get("optimized_title", "")
    if not title:
        logger.warning("No title found in metadata, using default")
        title = f"Video {validated.get('video_index', 'Unknown')}"

    # Get description
    description = validated.get("description", "") or validated.get("optimized_description", "")

    # Get tags
    tags = validated.get("tags", []) or validated.get("optimized_tags", [])

    # Validate description and tags using YouTube limits
    validated_description, desc_warnings = validate_description(
        description,
        DEFAULT_YOUTUBE_DESCRIPTION_LIMIT
    )

    validated_tags, tag_warnings = validate_tags(
        tags,
        DEFAULT_YOUTUBE_TAG_LIMIT,
        DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT,
        DEFAULT_YOUTUBE_MAX_TAGS_COUNT
    )

    # Log warnings
    for warning in desc_warnings:
        logger.warning(f"Description validation: {warning}")

    for warning in tag_warnings:
        logger.warning(f"Tags validation: {warning}")

    # Update metadata with validated values
    validated["title"] = title
    validated["description"] = validated_description
    validated["tags"] = validated_tags

    return validated

def upload_video_pom(
    driver: webdriver.Firefox,
    video_file: str,
    metadata: Dict[str, Any],
    publish_now: bool = True,
    schedule_time: Optional[datetime] = None,
    ffmpeg_path: Optional[str] = None,
    enable_recording: bool = False
) -> Optional[str]:
    """
    Upload a video to YouTube using the Page Object Model.

    Args:
        driver: WebDriver instance
        video_file: Path to video file
        metadata: Video metadata
        publish_now: Whether to publish immediately
        schedule_time: Target date/time to schedule (if not publishing now)
        ffmpeg_path: Path to FFmpeg executable (for recording)
        enable_recording: Whether to enable debug recording

    Returns:
        YouTube video ID if upload was successful, None otherwise
    """
    # Get video index for logging
    video_index = metadata.get("video_index", "UNKNOWN")
    logger.info(f"Starting upload process for video {video_index}: {video_file}")

    # Validate metadata
    validated_metadata = validate_metadata(metadata)

    # Start recording if enabled
    recording_process = None
    recording_filename = None

    if enable_recording and ffmpeg_path:
        recording_result = start_recording(str(video_index), ffmpeg_path, driver)
        if recording_result:
            recording_process, recording_filename = recording_result

    try:
        # 1. Navigate to YouTube Studio
        studio_page = StudioHomePage(driver)
        if not studio_page.navigate():
            logger.error("Failed to navigate to YouTube Studio")
            return None

        # 2. Start upload process
        if not studio_page.start_upload_process():
            logger.error("Failed to start upload process")
            return None

        # 3. Select file
        upload_page = UploadPage(driver)
        if not upload_page.select_file(video_file):
            logger.error(f"Failed to select video file: {video_file}")
            return None

        # 4. Wait for upload to start
        if upload_page.wait_for_upload_progress() is None:
            logger.error("Upload did not start")
            return None

        # 5. Wait for upload to complete
        if not upload_page.wait_for_upload_complete():
            logger.error("Upload did not complete")
            return None

        # 6. Fill details
        details_page = DetailsPage(driver)
        if not details_page.fill_details(validated_metadata):
            logger.error("Failed to fill video details")
            return None

        # 7. Set visibility
        visibility_page = VisibilityPage(driver)
        if not visibility_page.set_visibility(publish_now, schedule_time):
            logger.error("Failed to set visibility options")
            return None

        # 8. Wait for confirmation and get video ID
        confirmation_page = ConfirmationPage(driver)
        if not confirmation_page.wait_for_confirmation():
            logger.error("Did not get upload confirmation")
            return None

        # 9. Extract video ID
        video_id = confirmation_page.get_video_id()
        if not video_id:
            logger.error("Could not extract YouTube video ID")
            return None

        # 10. Close confirmation dialog
        confirmation_page.close_confirmation()

        logger.info(f"Upload successful! Video ID: {video_id}")
        return video_id

    except Exception as e:
        logger.error(f"Error during upload process: {e}", exc_info=True)
        return None

    finally:
        # Stop recording if started
        if recording_process and recording_filename:
            # Keep recording only if upload failed (video_id is None)
            keep_recording = 'video_id' not in locals() or video_id is None
            stop_recording(recording_process, recording_filename, keep_recording)

def add_video_to_playlist(service: Any, video_id: str, playlist_id: str) -> bool:
    """
    Add a video to a YouTube playlist using the YouTube Data API.

    Args:
        service: Authenticated YouTube API service
        video_id: YouTube video ID
        playlist_id: YouTube playlist ID

    Returns:
        True if video was added successfully, False otherwise
    """
    if not service or not video_id or not playlist_id:
        logger.error("Missing required parameters for adding to playlist")
        return False

    try:
        from googleapiclient.errors import HttpError

        logger.info(f"Adding video {video_id} to playlist {playlist_id}")

        # Create request
        request = service.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        )

        # Execute request
        response = request.execute()
        logger.info(f"Successfully added video {video_id} to playlist {playlist_id}")
        return True

    except HttpError as e:
        # Handle common errors
        error_details = e.resp.get('content', b'').decode('utf-8')

        if e.resp.status == 404:
            if "playlistNotFound" in error_details:
                logger.warning(f"Playlist ID '{playlist_id}' not found")
            elif "videoNotFound" in error_details:
                logger.warning(f"Video ID '{video_id}' not found")
            else:
                logger.error(f"API Error 404 adding video to playlist: {e}")
        elif e.resp.status == 403:
            logger.error(f"API Error 403 (Permission denied) adding video to playlist: {e}")
        else:
            logger.error(f"API Error adding video to playlist: {e}")

        return False

    except Exception as e:
        logger.error(f"Unexpected error adding video to playlist: {e}")
        return False

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.txt file.

    Returns:
        Dictionary containing configuration values
    """
    config = {}
    config_path = constants.CONFIG_FILE_PATH if CONSTANTS_IMPORTED else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.txt")

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()

        logger.info(f"Loaded configuration with {len(config)} settings")
        return config

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def main():
    """Main function for testing the uploader."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YouTube Video Uploader with Page Object Model")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--title", help="Video title")
    parser.add_argument("--description", help="Video description")
    parser.add_argument("--tags", help="Comma-separated list of tags")
    parser.add_argument("--schedule", action="store_true", help="Schedule video instead of publishing immediately")
    parser.add_argument("--schedule-time", help="Schedule time (format: YYYY-MM-DD HH:MM)")
    parser.add_argument("--record", action="store_true", help="Enable debug recording")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Get Firefox profile path from config
    profile_path = config.get("PROFILE_PATH")

    # Get FFmpeg path from config
    ffmpeg_path = config.get("FFMPEG_PATH")

    # Set up browser
    driver = setup_browser(profile_path)
    if not driver:
        logger.error("Failed to set up browser. Exiting.")
        return

    try:
        # Prepare metadata
        metadata = {
            "title": args.title or os.path.splitext(os.path.basename(args.video))[0],
            "description": args.description or "",
            "tags": [tag.strip() for tag in args.tags.split(",")] if args.tags else [],
            "video_index": "CLI"
        }

        # Determine publish mode
        publish_now = not args.schedule
        schedule_time = None

        if args.schedule and args.schedule_time:
            try:
                schedule_time = datetime.strptime(args.schedule_time, "%Y-%m-%d %H:%M")
            except ValueError:
                logger.error(f"Invalid schedule time format: {args.schedule_time}. Expected format: YYYY-MM-DD HH:MM")
                return
        elif args.schedule:
            # Default to tomorrow at noon if no specific time provided
            schedule_time = datetime.now() + timedelta(days=1)
            schedule_time = schedule_time.replace(hour=12, minute=0, second=0, microsecond=0)

        # Upload video
        video_id = upload_video_pom(
            driver=driver,
            video_file=args.video,
            metadata=metadata,
            publish_now=publish_now,
            schedule_time=schedule_time,
            ffmpeg_path=ffmpeg_path,
            enable_recording=args.record
        )

        if video_id:
            logger.info(f"Upload successful! Video ID: {video_id}")
            print(f"\nUpload successful! Video ID: {video_id}")
            print(f"Video URL: https://youtu.be/{video_id}")
        else:
            logger.error("Upload failed.")
            print("\nUpload failed. Check the log for details.")

    finally:
        # Close browser
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()
