# YouTube Shorts Automation - TODO List

This file tracks planned enhancements and features for the YouTube Shorts Automation project.

## Planned Features

### Dynamic Scheduling Based on Performance (Advanced)
- **Concept**: Instead of fixed intervals or predefined custom times, adjust schedule density based on channel performance (requires external analytics data or heuristics).
- **Implementation**:
  - Integrate (manually or via API) data about your channel's peak viewer hours/days.
  - Modify the scheduling logic (default_interval or custom_tomorrow's fallback) to prioritize scheduling uploads closer to these peak times.
  - Could potentially schedule more frequently during peak times if the upload queue is long and performance is good.
- **Benefit**: Potentially maximizes initial video visibility by aligning uploads with audience activity.

## Completed Features

- Enhanced validation tracking and automated prompt improvement for metadata generation
- Fixed tag validation to account for implicit quotes in tags with spaces
