#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Video Selection Algorithm

This module provides an intelligent scoring system to select videos that are more
likely to perform well based on various factors including engagement metrics,
trending topics, and historical performance.

Copyright (c) 2023-2025 Shahid Ali
License: MIT License
GitHub: https://github.com/Mrshahidali420/youtube-shorts-automation
Version: 1.5.0
"""

import os
import json
import re
import math
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

# Try to import YouTube API libraries
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print("Warning: Google API libraries not found. Some features will be limited.")
    YOUTUBE_API_AVAILABLE = False

# --- Constants ---
CONFIG_FILENAME = "config.txt"
TRENDING_TOPICS_CACHE_FILE = "trending_topics_cache.json"
HISTORICAL_PERFORMANCE_FILE = "historical_performance.json"
VIDEO_SCORES_CACHE_FILE = "video_scores_cache.json"

# Scoring weights (can be adjusted)
WEIGHT_VIEWS = 0.3
WEIGHT_ENGAGEMENT = 0.3
WEIGHT_TRENDING = 0.2
WEIGHT_HISTORICAL = 0.2

# Trending topics refresh interval (in hours)
TRENDING_REFRESH_HOURS = 24

# --- Setup ---
script_directory = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(script_directory, CONFIG_FILENAME)
trending_topics_cache_path = os.path.join(script_directory, TRENDING_TOPICS_CACHE_FILE)
historical_performance_path = os.path.join(script_directory, HISTORICAL_PERFORMANCE_FILE)
video_scores_cache_path = os.path.join(script_directory, VIDEO_SCORES_CACHE_FILE)

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

# --- YouTube API Functions ---
def get_youtube_api_client() -> Optional[Any]:
    """Initialize and return a YouTube API client."""
    if not YOUTUBE_API_AVAILABLE:
        log_error("YouTube API libraries not available.")
        return None

    config = load_config()
    api_key = config.get("API_KEY") or config.get("YOUTUBE_API_KEY")

    if not api_key:
        log_error("API_KEY not found in config.txt")
        return None

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        return youtube
    except Exception as e:
        log_error(f"Error initializing YouTube API client: {e}")
        return None

def get_trending_topics() -> List[str]:
    """
    Get trending topics related to the channel's niche.
    Uses a cache to avoid excessive API calls.
    """
    # Check if we have a recent cache
    cache = load_json_file(trending_topics_cache_path, {"topics": [], "last_updated": ""})

    # If cache is recent, use it
    if cache.get("last_updated"):
        try:
            last_updated = datetime.fromisoformat(cache["last_updated"])
            if datetime.now() - last_updated < timedelta(hours=TRENDING_REFRESH_HOURS):
                log_info(f"Using cached trending topics (updated {last_updated.strftime('%Y-%m-%d %H:%M:%S')})")
                return cache.get("topics", [])
        except:
            pass

    # Get channel topic from config
    config = load_config()
    channel_topic = config.get("SEO_CHANNEL_TOPIC", "")

    if not channel_topic:
        log_warning("SEO_CHANNEL_TOPIC not found in config.txt")
        return []

    # Extract main topics from channel topic
    main_topics = [topic.strip() for topic in channel_topic.split(',')]

    # Get trending videos for each topic
    all_trending_topics = []
    youtube = get_youtube_api_client()

    if not youtube:
        return []

    for topic in main_topics:
        if not topic:
            continue

        try:
            # Search for trending videos related to the topic
            search_response = youtube.search().list(
                q=topic,
                part="snippet",
                maxResults=10,
                type="video",
                order="viewCount",
                publishedAfter=(datetime.now() - timedelta(days=7)).isoformat() + "Z"
            ).execute()

            # Extract topics from video titles and descriptions
            for item in search_response.get("items", []):
                snippet = item.get("snippet", {})
                title = snippet.get("title", "")
                description = snippet.get("description", "")

                # Extract potential topics (nouns and noun phrases)
                words = re.findall(r'\b[A-Za-z][A-Za-z\s]{2,}\b', title + " " + description)
                all_trending_topics.extend([word.strip().lower() for word in words if len(word.strip()) > 3])

            # Respect API quota
            time.sleep(1)

        except Exception as e:
            log_error(f"Error fetching trending videos for topic '{topic}': {e}")

    # Count occurrences and get top trending topics
    if all_trending_topics:
        counter = Counter(all_trending_topics)
        trending_topics = [topic for topic, count in counter.most_common(20)]

        # Update cache
        cache = {
            "topics": trending_topics,
            "last_updated": datetime.now().isoformat()
        }
        save_json_file(trending_topics_cache_path, cache)

        return trending_topics

    return []

# --- Historical Performance Analysis ---
def load_historical_performance() -> Dict[str, Any]:
    """Load historical performance data for content types."""
    return load_json_file(historical_performance_path, {
        "topic_performance": {},
        "keyword_performance": {},
        "last_updated": ""
    })

def update_historical_performance(video_data: Dict[str, Any]) -> None:
    """
    Update historical performance data based on new video data.

    Args:
        video_data: Dictionary containing video information and performance metrics
    """
    historical = load_historical_performance()

    # Extract topics and keywords from video title and tags
    title = video_data.get("title", "").lower()
    tags = [tag.lower() for tag in video_data.get("tags", [])]

    # Performance metrics
    views = video_data.get("views", 0)
    engagement = video_data.get("engagement_rate", 0)

    # Update topic performance
    topic_performance = historical.get("topic_performance", {})

    # Extract potential topics from title (simple approach)
    words = re.findall(r'\b[A-Za-z][A-Za-z\s]{2,}\b', title)
    potential_topics = [word.strip().lower() for word in words if len(word.strip()) > 3]

    for topic in potential_topics:
        if topic not in topic_performance:
            topic_performance[topic] = {
                "videos": 0,
                "avg_views": 0,
                "avg_engagement": 0
            }

        # Update running average
        current = topic_performance[topic]
        n = current["videos"]
        current["avg_views"] = (current["avg_views"] * n + views) / (n + 1)
        current["avg_engagement"] = (current["avg_engagement"] * n + engagement) / (n + 1)
        current["videos"] += 1

    # Update keyword performance
    keyword_performance = historical.get("keyword_performance", {})

    for tag in tags:
        if tag not in keyword_performance:
            keyword_performance[tag] = {
                "videos": 0,
                "avg_views": 0,
                "avg_engagement": 0
            }

        # Update running average
        current = keyword_performance[tag]
        n = current["videos"]
        current["avg_views"] = (current["avg_views"] * n + views) / (n + 1)
        current["avg_engagement"] = (current["avg_engagement"] * n + engagement) / (n + 1)
        current["videos"] += 1

    # Save updated historical performance
    historical["topic_performance"] = topic_performance
    historical["keyword_performance"] = keyword_performance
    historical["last_updated"] = datetime.now().isoformat()

    save_json_file(historical_performance_path, historical)

# --- Scoring Functions ---
def calculate_base_score(video_data: Dict[str, Any]) -> float:
    """
    Calculate the base score for a video based on views and engagement.

    Args:
        video_data: Dictionary containing video information

    Returns:
        Base score between 0 and 100
    """
    # Extract metrics
    views = video_data.get("view_count", 0)
    likes = video_data.get("like_count", 0)
    comments = video_data.get("comment_count", 0)

    # Calculate engagement rate
    engagement_rate = 0
    if views > 0:
        engagement_rate = (likes + comments) / views * 100

    # Normalize views (logarithmic scale to handle viral videos)
    # A video with 1M views gets a score of 100, 100K views gets ~83, 10K views gets ~67
    normalized_views = min(100, max(0, 100 * math.log10(max(1, views)) / 6))

    # Normalize engagement rate (linear scale with cap)
    # 10% engagement rate gets a score of 100
    normalized_engagement = min(100, max(0, engagement_rate * 10))

    # Combine scores using weights
    base_score = (normalized_views * WEIGHT_VIEWS) + (normalized_engagement * WEIGHT_ENGAGEMENT)

    return base_score

def calculate_trending_score(video_data: Dict[str, Any]) -> float:
    """
    Calculate a trending score based on how well the video matches current trending topics.

    Args:
        video_data: Dictionary containing video information

    Returns:
        Trending score between 0 and 100
    """
    # Get trending topics
    trending_topics = get_trending_topics()

    if not trending_topics:
        return 50  # Neutral score if no trending topics available

    # Extract video title and tags
    title = video_data.get("title", "").lower()
    description = video_data.get("description", "").lower()
    tags = [tag.lower() for tag in video_data.get("tags", [])]

    # Count matches with trending topics
    matches = 0
    for topic in trending_topics:
        if topic.lower() in title or topic.lower() in description or topic.lower() in " ".join(tags):
            matches += 1

    # Calculate score based on matches
    # If the video matches 3 or more trending topics, it gets a score of 100
    trending_score = min(100, max(0, matches * 33.33))

    return trending_score

def calculate_historical_score(video_data: Dict[str, Any]) -> float:
    """
    Calculate a score based on historical performance of similar content.

    Args:
        video_data: Dictionary containing video information

    Returns:
        Historical score between 0 and 100
    """
    historical = load_historical_performance()

    if not historical.get("topic_performance") and not historical.get("keyword_performance"):
        return 50  # Neutral score if no historical data available

    # Extract video title and tags
    title = video_data.get("title", "").lower()
    tags = [tag.lower() for tag in video_data.get("tags", [])]

    # Extract potential topics from title
    words = re.findall(r'\b[A-Za-z][A-Za-z\s]{2,}\b', title)
    potential_topics = [word.strip().lower() for word in words if len(word.strip()) > 3]

    # Get performance data for matching topics and tags
    topic_scores = []
    for topic in potential_topics:
        if topic in historical.get("topic_performance", {}):
            topic_data = historical["topic_performance"][topic]
            if topic_data["videos"] >= 3:  # Only consider topics with enough data
                # Calculate a score based on views and engagement
                views_score = min(100, topic_data["avg_views"] / 1000 * 10)  # 10K views = 100 points
                engagement_score = min(100, topic_data["avg_engagement"] * 10)  # 10% engagement = 100 points
                topic_scores.append((views_score + engagement_score) / 2)

    tag_scores = []
    for tag in tags:
        if tag in historical.get("keyword_performance", {}):
            tag_data = historical["keyword_performance"][tag]
            if tag_data["videos"] >= 3:  # Only consider tags with enough data
                # Calculate a score based on views and engagement
                views_score = min(100, tag_data["avg_views"] / 1000 * 10)  # 10K views = 100 points
                engagement_score = min(100, tag_data["avg_engagement"] * 10)  # 10% engagement = 100 points
                tag_scores.append((views_score + engagement_score) / 2)

    # Combine scores
    if topic_scores and tag_scores:
        # Weight topic scores more heavily than tag scores
        historical_score = (sum(topic_scores) * 0.7 + sum(tag_scores) * 0.3) / (len(topic_scores) * 0.7 + len(tag_scores) * 0.3)
    elif topic_scores:
        historical_score = sum(topic_scores) / len(topic_scores)
    elif tag_scores:
        historical_score = sum(tag_scores) / len(tag_scores)
    else:
        historical_score = 50  # Neutral score if no matches

    return min(100, max(0, historical_score))

def calculate_video_potential_score(video_data: Dict[str, Any]) -> float:
    """
    Calculate the overall potential score for a video.

    Args:
        video_data: Dictionary containing video information

    Returns:
        Overall potential score between 0 and 100
    """
    # Calculate component scores
    base_score = calculate_base_score(video_data)
    trending_score = calculate_trending_score(video_data)
    historical_score = calculate_historical_score(video_data)

    # Combine scores using weights
    potential_score = (
        base_score * (WEIGHT_VIEWS + WEIGHT_ENGAGEMENT) +
        trending_score * WEIGHT_TRENDING +
        historical_score * WEIGHT_HISTORICAL
    )

    # Normalize to 0-100 scale
    potential_score = min(100, max(0, potential_score))

    # Add score components to video data for reference
    video_data["score_components"] = {
        "base_score": base_score,
        "trending_score": trending_score,
        "historical_score": historical_score,
        "potential_score": potential_score
    }

    return potential_score

# --- Main Selection Functions ---
def score_videos(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score a list of videos based on their potential performance.

    Args:
        videos: List of video data dictionaries

    Returns:
        List of videos with added scores
    """
    scored_videos = []

    for video in videos:
        # Calculate potential score
        potential_score = calculate_video_potential_score(video)

        # Add score to video data
        video["potential_score"] = potential_score
        scored_videos.append(video)

    # Sort by potential score (descending)
    scored_videos.sort(key=lambda x: x.get("potential_score", 0), reverse=True)

    return scored_videos

def select_best_videos(videos: List[Dict[str, Any]], count: int = 10) -> List[Dict[str, Any]]:
    """
    Select the best videos from a list based on their potential scores.

    Args:
        videos: List of video data dictionaries
        count: Number of videos to select

    Returns:
        List of selected videos
    """
    # Score videos
    scored_videos = score_videos(videos)

    # Select top videos
    selected_videos = scored_videos[:count]

    # Log selection
    log_info(f"Selected {len(selected_videos)} videos from {len(videos)} candidates")
    for i, video in enumerate(selected_videos, 1):
        log_info(f"  {i}. Score: {video.get('potential_score', 0):.1f} - {video.get('title', 'Unknown')}")

    return selected_videos

def select_videos_with_diversity(videos: List[Dict[str, Any]], count: int = 10, diversity_factor: float = 0.2) -> List[Dict[str, Any]]:
    """
    Select videos with a balance between high scores and diversity.

    Args:
        videos: List of video data dictionaries
        count: Number of videos to select
        diversity_factor: How much to prioritize diversity (0-1)

    Returns:
        List of selected videos
    """
    if count >= len(videos):
        return videos

    # Score videos
    scored_videos = score_videos(videos)

    # Initialize selection
    selected_videos = [scored_videos[0]]  # Start with the highest-scoring video
    remaining_videos = scored_videos[1:]

    # Select remaining videos with diversity consideration
    while len(selected_videos) < count and remaining_videos:
        # Calculate diversity scores
        max_diversity_scores = []

        for candidate in remaining_videos:
            # Calculate how different this video is from already selected videos
            diversity_scores = []

            for selected in selected_videos:
                # Simple diversity metric: title word overlap
                candidate_words = set(re.findall(r'\b[A-Za-z]+\b', candidate.get("title", "").lower()))
                selected_words = set(re.findall(r'\b[A-Za-z]+\b', selected.get("title", "").lower()))

                if not candidate_words or not selected_words:
                    diversity_scores.append(0.5)  # Neutral score if no words
                else:
                    overlap = len(candidate_words.intersection(selected_words))
                    total = len(candidate_words.union(selected_words))
                    diversity_score = 1 - (overlap / total)  # Higher score = more diverse
                    diversity_scores.append(diversity_score)

            # Use the maximum diversity score (most different from any selected video)
            max_diversity_score = max(diversity_scores) if diversity_scores else 0.5
            max_diversity_scores.append(max_diversity_score)

        # Calculate combined scores
        combined_scores = []
        for i, candidate in enumerate(remaining_videos):
            potential_score = candidate.get("potential_score", 0) / 100  # Normalize to 0-1
            diversity_score = max_diversity_scores[i]

            # Combine scores with diversity factor
            combined_score = (1 - diversity_factor) * potential_score + diversity_factor * diversity_score
            combined_scores.append(combined_score)

        # Select the video with the highest combined score
        best_index = combined_scores.index(max(combined_scores))
        selected_videos.append(remaining_videos[best_index])
        remaining_videos.pop(best_index)

    # Log selection
    log_info(f"Selected {len(selected_videos)} diverse videos from {len(videos)} candidates")
    for i, video in enumerate(selected_videos, 1):
        log_info(f"  {i}. Score: {video.get('potential_score', 0):.1f} - {video.get('title', 'Unknown')}")

    return selected_videos

# --- Integration Functions ---
def preprocess_yt_dlp_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess video entries from yt-dlp to standardize format for scoring.

    Args:
        entries: List of video entries from yt-dlp

    Returns:
        List of preprocessed video data dictionaries
    """
    preprocessed = []

    for entry in entries:
        # Extract relevant information
        video_id = entry.get("id", "")
        title = entry.get("title", "")
        description = entry.get("description", "")
        view_count = entry.get("view_count", 0)
        like_count = entry.get("like_count", 0)
        comment_count = entry.get("comment_count", 0)
        tags = entry.get("tags", [])
        duration = entry.get("duration", 0)

        # Create standardized video data
        video_data = {
            "video_id": video_id,
            "title": title,
            "description": description,
            "view_count": view_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "tags": tags,
            "duration": duration,
            # Add any other relevant fields
        }

        preprocessed.append(video_data)

    return preprocessed

# --- Main Function ---
def main():
    """Main function for testing."""
    log_info("Smart Video Selection Algorithm")
    log_info("-------------------------------")

    # Example usage
    example_videos = [
        {
            "video_id": "video1",
            "title": "GTA 6 Trailer Analysis - Hidden Details",
            "description": "Analyzing the hidden details in the GTA 6 trailer that you might have missed.",
            "view_count": 50000,
            "like_count": 2500,
            "comment_count": 300,
            "tags": ["gta 6", "trailer", "analysis", "rockstar games", "grand theft auto"]
        },
        {
            "video_id": "video2",
            "title": "GTA Online Best Money Making Methods 2023",
            "description": "The best ways to make money in GTA Online in 2023.",
            "view_count": 30000,
            "like_count": 1800,
            "comment_count": 200,
            "tags": ["gta online", "money", "guide", "tips", "grand theft auto"]
        },
        {
            "video_id": "video3",
            "title": "Top 10 GTA 6 Features We Want to See",
            "description": "The top 10 features we hope to see in GTA 6 when it releases.",
            "view_count": 100000,
            "like_count": 7000,
            "comment_count": 800,
            "tags": ["gta 6", "features", "wishlist", "rockstar games", "grand theft auto"]
        }
    ]

    # Score and select videos
    selected_videos = select_best_videos(example_videos, 2)

    # Select with diversity
    diverse_videos = select_videos_with_diversity(example_videos, 2)

    log_info("Done!")

if __name__ == "__main__":
    main()
