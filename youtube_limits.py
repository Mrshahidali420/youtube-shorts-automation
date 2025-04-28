# youtube_limits.py

# --- YouTube Limits Constants (Defaults) ---
# These act as fallbacks if not provided by the calling script
DEFAULT_YOUTUBE_DESCRIPTION_LIMIT = 4090
DEFAULT_YOUTUBE_TAG_LIMIT = 100

# --- ADJUSTED LIMIT ---
# Reduced slightly to account for potential discrepancies in YT's counting
# Original was 470, try 460 first. If still over, try 450.
DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT = 460
# --- END ADJUSTED LIMIT ---

DEFAULT_YOUTUBE_MAX_TAGS_COUNT = 40
# --- End Constants ---


def validate_description(
    description: str,
    limit: int = DEFAULT_YOUTUBE_DESCRIPTION_LIMIT # Accept limit as argument
) -> tuple[str, list[str]]:
    """
    Validates and truncates a description to meet YouTube's character limit.

    Args:
        description: The original video description string.
        limit: The character limit to enforce.

    Returns:
        A tuple containing:
            - The validated (potentially truncated) description string.
            - A list of warning messages generated during validation.
    """
    warnings = []
    if not description:
        return "", warnings

    description = str(description) # Ensure string type

    if len(description) > limit: # Use the passed 'limit' argument
        warnings.append(f"Description length ({len(description)}) exceeds limit ({limit}), truncated.")

        # Find the last space within the truncated limit
        truncated = description[:limit]
        last_space = truncated.rfind(' ')

        # Truncate at the last space only if it's reasonably close to the end
        # to avoid cutting off too much for a very long word at the boundary.
        if last_space > limit * 0.9: # Example: If last space is within the last 10%
             validated_description = truncated[:last_space].strip()
        else:
             # Otherwise, just cut at the limit
             validated_description = truncated.strip()

        return validated_description, warnings

    # Return original if within limit
    return description, warnings


def validate_tags(
    tags: list,
    tag_char_limit: int = DEFAULT_YOUTUBE_TAG_LIMIT,        # Accept limits as arguments
    total_char_limit: int = DEFAULT_YOUTUBE_TOTAL_TAGS_LIMIT, # Uses the (potentially adjusted) default
    max_count_limit: int = DEFAULT_YOUTUBE_MAX_TAGS_COUNT
) -> tuple[list[str], list[str]]:
    """
    Validates and optimizes a list of tags to meet YouTube's limits.

    Args:
        tags: A list of tag strings.
        tag_char_limit: Max characters allowed per individual tag.
        total_char_limit: Max total characters allowed for all tags combined.
        max_count_limit: Max number of tags allowed.

    Returns:
        A tuple containing:
            - The validated list of tag strings.
            - A list of warning messages generated during validation.
    """
    warnings = []
    if not tags or not isinstance(tags, list):
        return [], warnings

    # --- Step 1: Clean and Normalize Tags ---
    cleaned_tags = []
    # Use a set to efficiently track unique cleaned tags during initial pass
    seen_tags = set()
    for i, tag in enumerate(tags):
        if not tag or not isinstance(tag, str) or not tag.strip():
            # warnings.append(f"Skipping empty or invalid tag at index {i}: '{tag}'") # Optional warning
            continue

        clean_tag = tag.strip() # Remove leading/trailing whitespace

        # Optional: Convert to lowercase to treat 'Tag' and 'tag' as duplicates?
        # clean_tag_lower = clean_tag.lower()
        # if clean_tag_lower not in seen_tags:
        #      cleaned_tags.append(clean_tag) # Add original case version
        #      seen_tags.add(clean_tag_lower)

        # If case matters and exact duplicates should be removed:
        if clean_tag and clean_tag not in seen_tags:
            cleaned_tags.append(clean_tag)
            seen_tags.add(clean_tag) # Add cleaned tag to prevent exact duplicates later
    # --- End Cleaning ---

    # --- Step 2: Apply Limits (using passed arguments) ---
    valid_tags = []
    total_chars = 0
    tag_count = 0
    limits_hit = False
    processed_tags = set() # Keep track of tags already added to final list

    for tag in cleaned_tags: # Iterate through the cleaned, unique-ish list
        if limits_hit: break

        # Prevent adding the same tag twice (e.g., if cleaning didn't catch case variations earlier)
        if tag in processed_tags:
            continue

        # 2a. Check Max Tag Count
        if tag_count >= max_count_limit:
            if not limits_hit: warnings.append(f"Max tag count ({max_count_limit}) reached. Remaining tags skipped.")
            limits_hit = True
            continue

        # 2b. Check/Truncate Individual Tag Length BEFORE length check
        original_tag_repr = f"'{tag[:30]}...'" if len(tag) > 30 else f"'{tag}'"
        if len(tag) > tag_char_limit:
            tag = tag[:tag_char_limit].strip() # Truncate and strip again
            warnings.append(f"Tag {original_tag_repr} truncated to '{tag}' (>{tag_char_limit} chars).")

        current_tag_len = len(tag)
        # Skip if tag became empty after truncation/stripping
        if current_tag_len == 0: continue

        # Re-check for duplicates after potential truncation
        if tag in processed_tags:
            continue

        # 2c. Check Total Character Count (using the standard calculation for now)
        # This calculation assumes a comma separator between tags.
        separator_len = 1 if valid_tags else 0 # Add 1 char for comma AFTER the first tag
        prospective_total = total_chars + current_tag_len + separator_len

        if prospective_total > total_char_limit: # Use the potentially reduced limit
            if not limits_hit: warnings.append(f"Total tag char limit (~{total_char_limit}) reached. Remaining tags skipped.")
            limits_hit = True
            continue

        # --- Add the tag ---
        valid_tags.append(tag)
        processed_tags.add(tag) # Mark as processed
        total_chars = prospective_total # Update total chars correctly
        tag_count += 1

    # --- End Applying Limits ---

    if limits_hit:
        print(f"[youtube_limits] Validation resulted in {len(valid_tags)} tags, {total_chars} total chars (approx). Limits enforced.")
    else:
        print(f"[youtube_limits] Validation resulted in {len(valid_tags)} tags, {total_chars} total chars (approx). No limits hit.")


    return valid_tags, warnings