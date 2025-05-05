#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Content Calendar

This module provides an automated content calendar system that schedules uploads
based on audience activity patterns, trending events, and optimal posting frequency.

Copyright (c) 2023-2025 Shahid Ali
License: MIT License
GitHub: https://github.com/Mrshahidali420/youtube-shorts-automation
Version: 1.5.0
"""

import os
import json
import time
import calendar
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import traceback

# --- Constants ---
CONFIG_FILENAME = "config.txt"
EXCEL_FILENAME = "shorts_data.xlsx"
CALENDAR_DATA_FILE = "content_calendar_data.json"
PERFORMANCE_HISTORY_FILE = "analytics_data/performance_history.json"

# Default time slots (24-hour format)
DEFAULT_TIME_SLOTS = [
    (8, 0),   # 8:00 AM
    (12, 0),  # 12:00 PM
    (15, 0),  # 3:00 PM
    (18, 0),  # 6:00 PM
    (20, 0),  # 8:00 PM
    (22, 0)   # 10:00 PM
]

# --- Setup ---
script_directory = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(script_directory, CONFIG_FILENAME)
excel_file_path = os.path.join(script_directory, EXCEL_FILENAME)
calendar_data_path = os.path.join(script_directory, CALENDAR_DATA_FILE)
performance_history_path = os.path.join(script_directory, PERFORMANCE_HISTORY_FILE)

# --- Logging Functions ---
def log_info(msg: str) -> None:
    """Log an informational message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] INFO: {msg}")

def log_warning(msg: str) -> None:
    """Log a warning message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] WARNING: {msg}")

def log_error(msg: str) -> None:
    """Log an error message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {msg}")

# --- Configuration Loading ---
def load_config() -> Dict[str, str]:
    """Load configuration from config.txt file."""
    config = {}
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        log_error(f"Error loading config: {e}")
        return {}

# --- Data Loading/Saving Functions ---
def load_json_file(file_path: str, default_value: Any = None) -> Any:
    """Load data from a JSON file with error handling."""
    if default_value is None:
        default_value = {}

    if not os.path.exists(file_path):
        return default_value

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Error loading file {file_path}: {e}")
        return default_value

def save_json_file(file_path: str, data: Any) -> bool:
    """Save data to a JSON file with error handling."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        log_error(f"Error saving file {file_path}: {e}")
        return False

def load_calendar_data() -> Dict[str, Any]:
    """Load content calendar data."""
    default_data = {
        "scheduled_uploads": [],
        "optimal_days": {},
        "optimal_times": {},
        "last_updated": "",
        "upload_history": []
    }
    return load_json_file(calendar_data_path, default_data)

def save_calendar_data(data: Dict[str, Any]) -> bool:
    """Save content calendar data."""
    data["last_updated"] = datetime.now().isoformat()
    return save_json_file(calendar_data_path, data)

# --- Performance Analysis Functions ---
def analyze_performance_data() -> Tuple[Dict[str, float], Dict[int, float]]:
    """
    Analyze performance data to determine optimal upload days and times.

    Returns:
        Tuple of (optimal_days, optimal_times) dictionaries
    """
    # Load performance history
    performance_data = load_json_file(performance_history_path)
    videos = performance_data.get("videos", [])

    if not videos:
        log_warning("No performance data available for analysis.")
        return {}, {}

    # Initialize counters
    day_performance = {day: {"count": 0, "views": 0, "engagement": 0} for day in calendar.day_name}
    hour_performance = {hour: {"count": 0, "views": 0, "engagement": 0} for hour in range(24)}

    # Analyze each video
    for video in videos:
        upload_date_str = video.get("upload_date", "")
        stats_history = video.get("stats_history", [])

        if not upload_date_str or not stats_history:
            continue

        try:
            # Parse upload date
            upload_date = datetime.fromisoformat(upload_date_str.split("+")[0])
            day_name = upload_date.strftime("%A")
            hour = upload_date.hour

            # Get the latest stats
            latest_stats = stats_history[-1]
            views = latest_stats.get("views", 0)
            likes = latest_stats.get("likes", 0)
            comments = latest_stats.get("comments", 0)

            # Calculate engagement rate
            engagement_rate = (likes + comments) / max(1, views) * 100

            # Update day performance
            day_performance[day_name]["count"] += 1
            day_performance[day_name]["views"] += views
            day_performance[day_name]["engagement"] += engagement_rate

            # Update hour performance
            hour_performance[hour]["count"] += 1
            hour_performance[hour]["views"] += views
            hour_performance[hour]["engagement"] += engagement_rate

        except (ValueError, KeyError, IndexError) as e:
            log_warning(f"Error processing video performance data: {e}")

    # Calculate average performance by day
    optimal_days = {}
    for day, data in day_performance.items():
        if data["count"] > 0:
            avg_views = data["views"] / data["count"]
            avg_engagement = data["engagement"] / data["count"]
            # Combined score (50% views, 50% engagement)
            optimal_days[day] = (avg_views * 0.5) + (avg_engagement * 0.5)

    # Calculate average performance by hour
    optimal_times = {}
    for hour, data in hour_performance.items():
        if data["count"] > 0:
            avg_views = data["views"] / data["count"]
            avg_engagement = data["engagement"] / data["count"]
            # Combined score (50% views, 50% engagement)
            optimal_times[hour] = (avg_views * 0.5) + (avg_engagement * 0.5)

    return optimal_days, optimal_times

# --- Calendar Functions ---
def get_optimal_schedule(days_ahead: int = 7, uploads_per_day: int = 2) -> List[datetime]:
    """
    Generate an optimal upload schedule based on performance data.

    Args:
        days_ahead: Number of days to schedule ahead
        uploads_per_day: Number of uploads per day

    Returns:
        List of datetime objects representing optimal upload times
    """
    # Load calendar data
    calendar_data = load_calendar_data()

    # Get optimal days and times from calendar data or analyze performance
    optimal_days = calendar_data.get("optimal_days", {})
    optimal_times = calendar_data.get("optimal_times", {})

    # If no optimal data exists, analyze performance
    if not optimal_days or not optimal_times:
        optimal_days, optimal_times = analyze_performance_data()

        # If still no data, use defaults
        if not optimal_days:
            optimal_days = {day: 1.0 for day in calendar.day_name}
        if not optimal_times:
            optimal_times = {hour: 1.0 for hour in range(8, 23)}  # 8 AM to 10 PM

        # Save to calendar data
        calendar_data["optimal_days"] = optimal_days
        calendar_data["optimal_times"] = optimal_times
        save_calendar_data(calendar_data)

    # Convert optimal_days to a list of (day_name, score) tuples and sort by score
    day_scores = [(day, score) for day, score in optimal_days.items()]
    day_scores.sort(key=lambda x: x[1], reverse=True)

    # Convert optimal_times to a list of (hour, score) tuples and sort by score
    time_scores = [(hour, score) for hour, score in optimal_times.items()]
    time_scores.sort(key=lambda x: x[1], reverse=True)

    # Get top hours
    top_hours = [hour for hour, _ in time_scores[:uploads_per_day * 2]]
    if not top_hours:
        # Use default time slots if no optimal times
        top_hours = [slot[0] for slot in DEFAULT_TIME_SLOTS]

    # Generate schedule
    now = datetime.now()
    schedule = []

    for day_offset in range(1, days_ahead + 1):
        target_date = now + timedelta(days=day_offset)
        day_name = target_date.strftime("%A")

        # Skip days with low scores (bottom 20%)
        day_score = optimal_days.get(day_name, 0)
        if day_score < sorted(optimal_days.values())[int(len(optimal_days) * 0.2)]:
            continue

        # Select random hours from top hours for this day
        day_hours = random.sample(top_hours, min(uploads_per_day, len(top_hours)))

        for hour in sorted(day_hours):
            # Create datetime for this slot
            slot_time = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            schedule.append(slot_time)

    return schedule

def get_next_available_slot() -> Optional[datetime]:
    """
    Get the next available upload slot from the schedule.

    Returns:
        datetime object for the next available slot, or None if no slots available
    """
    # Load calendar data
    calendar_data = load_calendar_data()
    scheduled_uploads = calendar_data.get("scheduled_uploads", [])

    # Convert string dates to datetime objects
    scheduled_times = []
    for upload in scheduled_uploads:
        try:
            scheduled_time = datetime.fromisoformat(upload["scheduled_time"])
            scheduled_times.append(scheduled_time)
        except (ValueError, KeyError):
            continue

    # Get optimal schedule
    optimal_schedule = get_optimal_schedule()

    # Find the next available slot
    now = datetime.now()
    for slot in sorted(optimal_schedule):
        if slot > now and slot not in scheduled_times:
            return slot

    # If no optimal slots available, create a fallback slot
    fallback_slot = now + timedelta(hours=1)
    fallback_slot = fallback_slot.replace(minute=0, second=0, microsecond=0)
    return fallback_slot

def schedule_upload(video_id: str, title: str, scheduled_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Schedule a video upload.

    Args:
        video_id: ID of the video to schedule
        title: Title of the video
        scheduled_time: Optional datetime to schedule the upload. If None, uses next available slot.

    Returns:
        Dictionary with scheduling information
    """
    if scheduled_time is None:
        scheduled_time = get_next_available_slot()

    # Create schedule entry
    schedule_entry = {
        "video_id": video_id,
        "title": title,
        "scheduled_time": scheduled_time.isoformat(),
        "created_at": datetime.now().isoformat()
    }

    # Load calendar data
    calendar_data = load_calendar_data()
    scheduled_uploads = calendar_data.get("scheduled_uploads", [])

    # Add to scheduled uploads
    scheduled_uploads.append(schedule_entry)
    calendar_data["scheduled_uploads"] = scheduled_uploads

    # Save calendar data
    save_calendar_data(calendar_data)

    return schedule_entry

def get_upcoming_schedule(days_ahead: int = 7) -> List[Dict[str, Any]]:
    """
    Get the upcoming upload schedule.

    Args:
        days_ahead: Number of days to look ahead

    Returns:
        List of scheduled uploads
    """
    # Load calendar data
    calendar_data = load_calendar_data()
    scheduled_uploads = calendar_data.get("scheduled_uploads", [])

    # Filter for upcoming uploads
    now = datetime.now()
    cutoff_date = now + timedelta(days=days_ahead)

    upcoming_uploads = []
    for upload in scheduled_uploads:
        try:
            scheduled_time = datetime.fromisoformat(upload["scheduled_time"])
            if now <= scheduled_time <= cutoff_date:
                upcoming_uploads.append({
                    "video_id": upload.get("video_id", ""),
                    "title": upload.get("title", ""),
                    "scheduled_time": scheduled_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "days_from_now": (scheduled_time - now).days,
                    "hours_from_now": round((scheduled_time - now).total_seconds() / 3600, 1)
                })
        except (ValueError, KeyError):
            continue

    # Sort by scheduled time
    upcoming_uploads.sort(key=lambda x: x["scheduled_time"])

    return upcoming_uploads

def mark_upload_complete(video_id: str, youtube_id: str = "") -> bool:
    """
    Mark a scheduled upload as complete.

    Args:
        video_id: ID of the video that was uploaded
        youtube_id: YouTube ID of the uploaded video

    Returns:
        True if successful, False otherwise
    """
    # Load calendar data
    calendar_data = load_calendar_data()
    scheduled_uploads = calendar_data.get("scheduled_uploads", [])
    upload_history = calendar_data.get("upload_history", [])

    # Find the scheduled upload
    found = False
    for i, upload in enumerate(scheduled_uploads):
        if upload.get("video_id") == video_id:
            # Create history entry
            history_entry = {
                "video_id": video_id,
                "youtube_id": youtube_id,
                "title": upload.get("title", ""),
                "scheduled_time": upload.get("scheduled_time", ""),
                "uploaded_time": datetime.now().isoformat()
            }

            # Add to history and remove from scheduled
            upload_history.append(history_entry)
            scheduled_uploads.pop(i)
            found = True
            break

    if found:
        # Update calendar data
        calendar_data["scheduled_uploads"] = scheduled_uploads
        calendar_data["upload_history"] = upload_history

        # Save calendar data
        return save_calendar_data(calendar_data)

    return False

def generate_calendar_report(days_ahead: int = 14) -> str:
    """
    Generate a human-readable calendar report.

    Args:
        days_ahead: Number of days to include in the report

    Returns:
        String containing the calendar report
    """
    # Get upcoming schedule
    upcoming_uploads = get_upcoming_schedule(days_ahead)

    # Generate report
    report = []
    report.append("=" * 60)
    report.append("CONTENT CALENDAR REPORT")
    report.append("=" * 60)
    report.append("")

    if not upcoming_uploads:
        report.append("No upcoming uploads scheduled.")
    else:
        # Group by date
        uploads_by_date = {}
        for upload in upcoming_uploads:
            scheduled_time = upload["scheduled_time"]
            date = scheduled_time.split(" ")[0]
            if date not in uploads_by_date:
                uploads_by_date[date] = []
            uploads_by_date[date].append(upload)

        # Generate report by date
        for date, uploads in sorted(uploads_by_date.items()):
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                day_name = date_obj.strftime("%A")
                report.append(f"{date} ({day_name}):")

                for upload in sorted(uploads, key=lambda x: x["scheduled_time"]):
                    time_str = upload["scheduled_time"].split(" ")[1]
                    title = upload["title"]
                    video_id = upload["video_id"]
                    report.append(f"  {time_str} - {title} (ID: {video_id})")

                report.append("")
            except ValueError:
                continue

    # Add optimal days and times
    calendar_data = load_calendar_data()
    optimal_days = calendar_data.get("optimal_days", {})
    optimal_times = calendar_data.get("optimal_times", {})

    if optimal_days:
        report.append("Optimal Upload Days (by performance score):")
        for day, score in sorted(optimal_days.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {day}: {score:.2f}")
        report.append("")

    if optimal_times:
        report.append("Optimal Upload Times (by performance score):")
        for hour, score in sorted(optimal_times.items(), key=lambda x: x[1], reverse=True)[:6]:
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12
            if display_hour == 0:
                display_hour = 12
            report.append(f"  {display_hour}:00 {am_pm}: {score:.2f}")
        report.append("")

    return "\n".join(report)

# --- Main Function ---
def main():
    """Main function for testing."""
    print("Automated Content Calendar")
    print("-------------------------")

    # Analyze performance data
    print("Analyzing performance data...")
    optimal_days, optimal_times = analyze_performance_data()

    print("\nOptimal Upload Days:")
    for day, score in sorted(optimal_days.items(), key=lambda x: x[1], reverse=True):
        print(f"  {day}: {score:.2f}")

    print("\nOptimal Upload Times:")
    for hour, score in sorted(optimal_times.items(), key=lambda x: x[1], reverse=True)[:6]:
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        print(f"  {display_hour}:00 {am_pm}: {score:.2f}")

    # Generate optimal schedule
    print("\nGenerating optimal schedule...")
    schedule = get_optimal_schedule()

    print("\nOptimal Upload Schedule:")
    for slot in schedule:
        print(f"  {slot.strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate calendar report
    print("\nGenerating calendar report...")
    report = generate_calendar_report()
    print("\n" + report)

    print("\nDone!")

if __name__ == "__main__":
    main()
