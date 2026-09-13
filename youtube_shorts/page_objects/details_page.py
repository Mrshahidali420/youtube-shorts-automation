"""
YouTube Studio Details Page Object

This module provides the Page Object for the YouTube Studio details page
where title, description, and tags are entered.

Copyright (c) 2023-2025 Shahid Ali
License: MIT License
GitHub: https://github.com/Mrshahidali420/youtube-shorts-automation
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .base_page import BasePage

class DetailsPage(BasePage):
    """Page Object for YouTube Studio details page."""
    
    # Locators
    TITLE_INPUT_LOCATORS = [
        (By.XPATH, "//ytcp-mention-textbox[@label='Title']//div[@contenteditable='true']"),
        (By.XPATH, "//div[@aria-label='Add a title that describes your video']"),
        (By.CSS_SELECTOR, "ytcp-mention-textbox[label='Title'] div[contenteditable='true']")
    ]
    
    DESCRIPTION_INPUT_LOCATORS = [
        (By.XPATH, "//ytcp-mention-textbox[@label='Description']//div[@contenteditable='true']"),
        (By.XPATH, "//div[@aria-label='Tell viewers about your video']"),
        (By.CSS_SELECTOR, "ytcp-mention-textbox[label='Description'] div[contenteditable='true']")
    ]
    
    SHOW_MORE_BUTTON_LOCATORS = [
        (By.XPATH, "//ytcp-button[@id='toggle-button']"),
        (By.XPATH, "//div[contains(text(), 'Show more')]"),
        (By.CSS_SELECTOR, "ytcp-button#toggle-button")
    ]
    
    TAGS_INPUT_LOCATORS = [
        (By.XPATH, "//input[@placeholder='Add tag']"),
        (By.XPATH, "//ytcp-chip-bar[@id='tags-container']//input"),
        (By.CSS_SELECTOR, "ytcp-chip-bar#tags-container input")
    ]
    
    NEXT_BUTTON_LOCATORS = [
        (By.XPATH, "//ytcp-button[@id='next-button']"),
        (By.XPATH, "//div[contains(@class, 'next-button')]"),
        (By.CSS_SELECTOR, "ytcp-button#next-button")
    ]
    
    THUMBNAIL_SECTION_LOCATORS = [
        (By.XPATH, "//h2[contains(text(), 'Thumbnail')]"),
        (By.XPATH, "//div[contains(@class, 'thumbnail-section')]"),
        (By.CSS_SELECTOR, "div.thumbnail-section")
    ]
    
    def __init__(self, driver: webdriver.Firefox):
        """Initialize the Details Page object."""
        super().__init__(driver, log_prefix="DetailsPage")
    
    def wait_for_page_to_load(self, timeout: int = 60) -> bool:
        """
        Wait for the details page to load.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if page loaded successfully, False otherwise
        """
        self.log_info("Waiting for details page to load...")
        
        # Wait for title input to be present
        title_input = self.find_element_with_multiple_locators(
            self.TITLE_INPUT_LOCATORS,
            wait=WebDriverWait(self.driver, timeout),
            take_screenshot_on_failure=True
        )
        
        if title_input:
            self.log_success("Details page loaded successfully")
            return True
        else:
            self.log_error("Failed to load details page")
            return False
    
    def fill_title(self, title: str) -> bool:
        """
        Fill in the video title.
        
        Args:
            title: Video title
            
        Returns:
            True if title was filled successfully, False otherwise
        """
        self.log_info(f"Filling title: {title}")
        
        title_input = self.find_element_with_multiple_locators(
            self.TITLE_INPUT_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        return self.enter_text(title_input, title, "title field")
    
    def fill_description(self, description: str) -> bool:
        """
        Fill in the video description.
        
        Args:
            description: Video description
            
        Returns:
            True if description was filled successfully, False otherwise
        """
        self.log_info(f"Filling description (length: {len(description)} chars)")
        
        description_input = self.find_element_with_multiple_locators(
            self.DESCRIPTION_INPUT_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        return self.enter_text(description_input, description, "description field")
    
    def click_show_more(self) -> bool:
        """
        Click the Show More button to reveal additional fields.
        
        Returns:
            True if click was successful, False otherwise
        """
        self.log_info("Clicking Show More button...")
        
        show_more_button = self.find_element_with_multiple_locators(
            self.SHOW_MORE_BUTTON_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        result = self.click_element_safely(show_more_button, "Show More button")
        if result:
            self.mimic_human_delay(0.5, 1.0)
        return result
    
    def fill_tags(self, tags: List[str]) -> bool:
        """
        Fill in the video tags.
        
        Args:
            tags: List of tags
            
        Returns:
            True if tags were filled successfully, False otherwise
        """
        if not tags:
            self.log_info("No tags to fill")
            return True
        
        self.log_info(f"Filling {len(tags)} tags")
        
        # First make sure the tags section is visible by clicking Show More
        self.click_show_more()
        
        # Find tags input
        tags_input = self.find_element_with_multiple_locators(
            self.TAGS_INPUT_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        if not tags_input:
            self.log_error("Tags input field not found")
            return False
        
        try:
            # Click the tags input
            self.click_element_safely(tags_input, "tags input field")
            
            # Enter each tag followed by comma
            for tag in tags:
                if not tag:
                    continue
                
                self.log_info(f"Adding tag: {tag}", indent=1)
                tags_input.send_keys(tag)
                tags_input.send_keys(Keys.COMMA)
                self.mimic_human_delay(0.1, 0.3)
            
            self.log_success(f"Successfully added {len(tags)} tags")
            return True
        except Exception as e:
            self.log_error(f"Failed to add tags: {e}")
            self.take_screenshot("tags_input_failed")
            return False
    
    def click_next(self) -> bool:
        """
        Click the Next button to proceed to the next page.
        
        Returns:
            True if click was successful, False otherwise
        """
        self.log_info("Clicking Next button...")
        
        next_button = self.find_element_with_multiple_locators(
            self.NEXT_BUTTON_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        result = self.click_element_safely(next_button, "Next button")
        if result:
            self.mimic_human_delay(0.5, 1.0)
        return result
    
    def fill_details(self, metadata: Dict[str, Any]) -> bool:
        """
        Fill in all details from metadata.
        
        Args:
            metadata: Dictionary containing title, description, and tags
            
        Returns:
            True if all details were filled successfully, False otherwise
        """
        self.log_info("Filling video details from metadata...")
        
        # Wait for page to load
        if not self.wait_for_page_to_load():
            return False
        
        # Fill title
        title = metadata.get("title", "") or metadata.get("optimized_title", "")
        if not self.fill_title(title):
            self.log_error("Failed to fill title")
            return False
        
        # Fill description
        description = metadata.get("description", "") or metadata.get("optimized_description", "")
        if not self.fill_description(description):
            self.log_error("Failed to fill description")
            return False
        
        # Fill tags
        tags = metadata.get("tags", []) or metadata.get("optimized_tags", [])
        if not self.fill_tags(tags):
            self.log_warning("Failed to fill tags, but continuing")
            # Don't return False here, as tags are not critical
        
        # Click Next
        if not self.click_next():
            self.log_error("Failed to click Next button")
            return False
        
        self.log_success("Successfully filled all details and proceeded to next page")
        return True
