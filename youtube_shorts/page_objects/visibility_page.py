"""
YouTube Studio Visibility Page Object

This module provides the Page Object for the YouTube Studio visibility page
where publishing options (public, private, scheduled) are set.

Copyright (c) 2023-2025 Shahid Ali
License: MIT License
GitHub: https://github.com/Mrshahidali420/youtube-shorts-automation
Version: 1.0.0
"""

import time
from datetime import datetime
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .base_page import BasePage

class VisibilityPage(BasePage):
    """Page Object for YouTube Studio visibility page."""
    
    # Locators
    PUBLIC_RADIO_LOCATORS = [
        (By.XPATH, "//tp-yt-paper-radio-button[@name='PUBLIC']"),
        (By.XPATH, "//div[@id='visibility-list']//tp-yt-paper-radio-button[contains(., 'Public')]"),
        (By.CSS_SELECTOR, "tp-yt-paper-radio-button[name='PUBLIC']")
    ]
    
    SCHEDULE_RADIO_LOCATORS = [
        (By.XPATH, "//tp-yt-paper-radio-button[@name='SCHEDULED']"),
        (By.XPATH, "//div[@id='visibility-list']//tp-yt-paper-radio-button[contains(., 'Schedule')]"),
        (By.CSS_SELECTOR, "tp-yt-paper-radio-button[name='SCHEDULED']")
    ]
    
    DATE_PICKER_LOCATORS = [
        (By.XPATH, "//ytcp-date-picker"),
        (By.XPATH, "//input[contains(@placeholder, 'Date')]"),
        (By.CSS_SELECTOR, "ytcp-date-picker input")
    ]
    
    TIME_PICKER_LOCATORS = [
        (By.XPATH, "//ytcp-time-of-day-picker"),
        (By.XPATH, "//input[contains(@placeholder, 'Time')]"),
        (By.CSS_SELECTOR, "ytcp-time-of-day-picker input")
    ]
    
    PUBLISH_BUTTON_LOCATORS = [
        (By.XPATH, "//ytcp-button[@id='done-button']"),
        (By.XPATH, "//div[contains(@class, 'done-button')]"),
        (By.CSS_SELECTOR, "ytcp-button#done-button")
    ]
    
    SCHEDULE_BUTTON_LOCATORS = [
        (By.XPATH, "//ytcp-button[@id='done-button']"),
        (By.XPATH, "//div[contains(@class, 'done-button')]"),
        (By.CSS_SELECTOR, "ytcp-button#done-button")
    ]
    
    # Calendar locators
    CALENDAR_CONTAINER_LOCATORS = [
        (By.XPATH, "//tp-yt-paper-dialog[contains(@class, 'date-picker-dialog')]"),
        (By.CSS_SELECTOR, "tp-yt-paper-dialog.date-picker-dialog")
    ]
    
    CALENDAR_MONTH_YEAR_LOCATORS = [
        (By.XPATH, "//div[contains(@class, 'date-picker-header')]"),
        (By.CSS_SELECTOR, "div.date-picker-header")
    ]
    
    CALENDAR_NEXT_MONTH_LOCATORS = [
        (By.XPATH, "//iron-icon[@icon='chevron-right']"),
        (By.CSS_SELECTOR, "iron-icon[icon='chevron-right']")
    ]
    
    CALENDAR_DAY_LOCATORS = [
        (By.XPATH, "//div[contains(@class, 'date-picker-day')]"),
        (By.CSS_SELECTOR, "div.date-picker-day")
    ]
    
    CALENDAR_SAVE_BUTTON_LOCATORS = [
        (By.XPATH, "//div[contains(@class, 'date-picker-dialog')]//ytcp-button[contains(., 'Save')]"),
        (By.CSS_SELECTOR, "div.date-picker-dialog ytcp-button[dialog-confirm]")
    ]
    
    def __init__(self, driver: webdriver.Firefox):
        """Initialize the Visibility Page object."""
        super().__init__(driver, log_prefix="VisibilityPage")
    
    def wait_for_page_to_load(self, timeout: int = 60) -> bool:
        """
        Wait for the visibility page to load.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if page loaded successfully, False otherwise
        """
        self.log_info("Waiting for visibility page to load...")
        
        # Wait for public radio button to be present
        public_radio = self.find_element_with_multiple_locators(
            self.PUBLIC_RADIO_LOCATORS,
            wait=WebDriverWait(self.driver, timeout),
            take_screenshot_on_failure=True
        )
        
        if public_radio:
            self.log_success("Visibility page loaded successfully")
            return True
        else:
            self.log_error("Failed to load visibility page")
            return False
    
    def select_public(self) -> bool:
        """
        Select the Public radio button.
        
        Returns:
            True if selection was successful, False otherwise
        """
        self.log_info("Selecting Public visibility...")
        
        public_radio = self.find_element_with_multiple_locators(
            self.PUBLIC_RADIO_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        result = self.click_element_safely(public_radio, "Public radio button")
        if result:
            self.mimic_human_delay(0.5, 1.0)
        return result
    
    def select_schedule(self) -> bool:
        """
        Select the Schedule radio button.
        
        Returns:
            True if selection was successful, False otherwise
        """
        self.log_info("Selecting Schedule visibility...")
        
        schedule_radio = self.find_element_with_multiple_locators(
            self.SCHEDULE_RADIO_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        result = self.click_element_safely(schedule_radio, "Schedule radio button")
        if result:
            self.mimic_human_delay(0.5, 1.0)
        return result
    
    def set_date(self, target_date: datetime) -> bool:
        """
        Set the schedule date using the date picker.
        
        Args:
            target_date: Target date to schedule
            
        Returns:
            True if date was set successfully, False otherwise
        """
        self.log_info(f"Setting schedule date to {target_date.strftime('%Y-%m-%d')}")
        
        # Click the date picker to open the calendar
        date_picker = self.find_element_with_multiple_locators(
            self.DATE_PICKER_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        if not date_picker:
            self.log_error("Date picker not found")
            return False
        
        if not self.click_element_safely(date_picker, "date picker"):
            self.log_error("Failed to click date picker")
            return False
        
        # Wait for calendar to appear
        calendar = self.find_element_with_multiple_locators(
            self.CALENDAR_CONTAINER_LOCATORS,
            wait=self.wait_medium
        )
        
        if not calendar:
            self.log_error("Calendar not found after clicking date picker")
            return False
        
        # Get current month/year displayed in calendar
        month_year_element = self.find_element_with_multiple_locators(
            self.CALENDAR_MONTH_YEAR_LOCATORS,
            wait=self.wait_short
        )
        
        if not month_year_element:
            self.log_error("Month/year header not found in calendar")
            return False
        
        # Parse current month/year
        current_month_year = month_year_element.text
        self.log_info(f"Calendar showing: {current_month_year}")
        
        # Format target month/year
        target_month_year = target_date.strftime("%B %Y")
        
        # Navigate to target month if needed
        attempts = 0
        while current_month_year != target_month_year and attempts < 12:  # Limit to 12 months forward
            self.log_info(f"Navigating from {current_month_year} to {target_month_year}")
            
            # Click next month button
            next_month_button = self.find_element_with_multiple_locators(
                self.CALENDAR_NEXT_MONTH_LOCATORS,
                wait=self.wait_short,
                clickable=True
            )
            
            if not next_month_button:
                self.log_error("Next month button not found")
                return False
            
            self.click_element_safely(next_month_button, "next month button")
            self.mimic_human_delay(0.2, 0.5)
            
            # Get updated month/year
            month_year_element = self.find_element_with_multiple_locators(
                self.CALENDAR_MONTH_YEAR_LOCATORS,
                wait=self.wait_short
            )
            
            if month_year_element:
                current_month_year = month_year_element.text
            
            attempts += 1
        
        if attempts >= 12:
            self.log_error(f"Could not navigate to target month {target_month_year}")
            return False
        
        # Find the target day
        target_day = target_date.day
        day_elements = self.driver.find_elements(By.XPATH, f"//div[contains(@class, 'date-picker-day') and text()='{target_day}']")
        
        if not day_elements:
            self.log_error(f"Day {target_day} not found in calendar")
            return False
        
        # Click the day
        for day_element in day_elements:
            try:
                # Check if the day is selectable (not disabled)
                if "disabled" not in day_element.get_attribute("class"):
                    self.click_element_safely(day_element, f"day {target_day}")
                    break
            except Exception:
                continue
        else:
            self.log_error(f"No selectable day {target_day} found")
            return False
        
        # Click Save button
        save_button = self.find_element_with_multiple_locators(
            self.CALENDAR_SAVE_BUTTON_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        if not save_button:
            self.log_error("Save button not found in calendar")
            return False
        
        result = self.click_element_safely(save_button, "calendar save button")
        if result:
            self.mimic_human_delay(0.5, 1.0)
        
        return result
    
    def set_time(self, target_date: datetime) -> bool:
        """
        Set the schedule time.
        
        Args:
            target_date: Target date/time to schedule
            
        Returns:
            True if time was set successfully, False otherwise
        """
        self.log_info(f"Setting schedule time to {target_date.strftime('%H:%M')}")
        
        # Format time string (24-hour format)
        time_str = target_date.strftime("%H:%M")
        
        # Find time picker
        time_picker = self.find_element_with_multiple_locators(
            self.TIME_PICKER_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        if not time_picker:
            self.log_error("Time picker not found")
            return False
        
        # Clear and enter time
        return self.enter_text(time_picker, time_str, "time picker")
    
    def click_publish_button(self) -> bool:
        """
        Click the Publish button for public videos.
        
        Returns:
            True if click was successful, False otherwise
        """
        self.log_info("Clicking Publish button...")
        
        publish_button = self.find_element_with_multiple_locators(
            self.PUBLISH_BUTTON_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        result = self.click_element_safely(publish_button, "Publish button")
        if result:
            self.mimic_human_delay(1.0, 2.0)
        return result
    
    def click_schedule_button(self) -> bool:
        """
        Click the Schedule button for scheduled videos.
        
        Returns:
            True if click was successful, False otherwise
        """
        self.log_info("Clicking Schedule button...")
        
        schedule_button = self.find_element_with_multiple_locators(
            self.SCHEDULE_BUTTON_LOCATORS,
            wait=self.wait_medium,
            clickable=True
        )
        
        result = self.click_element_safely(schedule_button, "Schedule button")
        if result:
            self.mimic_human_delay(1.0, 2.0)
        return result
    
    def set_visibility(self, publish_now: bool, schedule_time: Optional[datetime] = None) -> bool:
        """
        Set the visibility options.
        
        Args:
            publish_now: Whether to publish immediately
            schedule_time: Target date/time to schedule (if not publishing now)
            
        Returns:
            True if visibility was set successfully, False otherwise
        """
        self.log_info(f"Setting visibility: publish_now={publish_now}, schedule_time={schedule_time}")
        
        # Wait for page to load
        if not self.wait_for_page_to_load():
            return False
        
        if publish_now:
            # Select Public and click Publish
            if not self.select_public():
                self.log_error("Failed to select Public visibility")
                return False
            
            if not self.click_publish_button():
                self.log_error("Failed to click Publish button")
                return False
        else:
            # Verify we have a schedule time
            if not schedule_time:
                self.log_error("Schedule time is required when not publishing immediately")
                return False
            
            # Select Schedule
            if not self.select_schedule():
                self.log_error("Failed to select Schedule visibility")
                return False
            
            # Set date and time
            if not self.set_date(schedule_time):
                self.log_error("Failed to set schedule date")
                return False
            
            if not self.set_time(schedule_time):
                self.log_error("Failed to set schedule time")
                return False
            
            # Click Schedule button
            if not self.click_schedule_button():
                self.log_error("Failed to click Schedule button")
                return False
        
        self.log_success("Successfully set visibility options")
        return True
