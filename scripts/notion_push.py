#!/usr/bin/env python3
"""
Vitals & Signals — Standalone Notion Push
==========================================
This script reads already-generated content from your local output/ folder
and pushes it to Notion as a new Draft page.

Use this when you want to:
  - Re-push content after editing local files (without re-running research)
  - Create a fresh Notion draft from files that already exist on disk
  - Test the Notion integration without running the full research pipeline

Usage:
  python scripts/notion_push.py               # push the most recent episode
  python scripts/notion_push.py --episode 3   # push a specific episode number

What it reads from output/:
  - research_brief.json   → episode title and meta-themes
  - blog_post.md          → newsletter content
  - podcast_script.txt    → podcast script

Dependencies (all in your venv):
  notion-client, python-dotenv
"""

import os
import re
import sys
import json
import argparse
import datetime
from pathlib import Path

# python-dotenv loads API keys from the .env file — no hardcoded secrets.
from dotenv import load_dotenv

# The official Notion Python SDK.
from notion_client import Client as NotionClient


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# Load secrets and set up the Notion client.
# ─────────────────────────────────────────────────────────────────────────────

# Load the .env file from the project root (one level up from /scripts/).
load_dotenv(Path(__file__).parent.parent / ".env")

# Pull Notion keys from environment. We don't need Anthropic here — no AI calls.
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DB_ID   = os.getenv("NOTION_DATABASE_ID")

# Fail early with a clear message if keys are missing.
missing = [k for k, v in {
    "NOTION_API_KEY":    NOTION_API_KEY,
    "NOTION_DATABASE_ID": NOTION_DB_ID,
}.items() if not v]
if missing:
    sys.exit(
        f"[ERROR] Missing required environment variables: {', '.join(missing)}\n"
        f"        Check your .env file in the project root."
    )

# Initialize the Notion client.
notion = NotionClient(auth=NOTION_API_KEY)

# Path to the output folder at the project root.
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Today's date — used to set a new publish date if one isn't stored.
TODAY = datetime.date.today()

# The publish date will be the next Monday after today.
days_until_monday = (7 - TODAY.weekday()) % 7 or 7
PUBLISH_DATE = TODAY + datetime.timedelta(days=days_until_monday)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: EPISODE DISCOVERY
# Find the right episode folder in output/.
# ─────────────────────────────────────────────────────────────────────────────

def find_episode_folder(episode_num: int | None) -> Path:
    """
    Finds the output folder for a given episode number.

    Episode folders follow this naming pattern: ep001-2024-01-15/
    (episode number padded to 3 digits, followed by the date it was created)

    If episode_num is given, we look for a folder starting with ep{N:03d}-.
    If not given, we return the most recently created episode folder.

    Args:
        episode_num: The episode number to find (e.g. 3), or None for the latest.

    Returns:
        Path to the matching episode folder.
    """
    if not OUTPUT_DIR.exists():
        sys.exit(
            f"[ERROR] Output directory not found: {OUTPUT_DIR}\n"
            f"        Run scripts/research.py first to generate content."
        )

    # List all subdirectories in output/ that match the ep### pattern.
    # Path.glob("ep*") finds all folders starting with "ep".
    episode_folders = sorted(OUTPUT_DIR.glob("ep*"), reverse=True)

    if not episode_folders:
        sys.exit(
            f"[ERROR] No episode folders found in {OUTPUT_DIR}\n"
            f"        Run scripts/research.py first to generate content."
        )

    if episode_num is not None:
        # Look for a folder whose name starts with the padded episode number.
        # e.g. episode 3 → looks for a folder starting with "ep003"
        prefix = f"ep{episode_num:03d}"
        matches = [f for f in episode_folders if f.name.startswith(prefix)]
        if not matches:
            available = [f.name for f in episode_folders]
            sys.exit(
                f"[ERROR] No folder found for episode {episode_num} (looking for {prefix}-).\n"
                f"        Available episodes: {', '.join(available)}"
            )
        return matches[0]

    # No episode number given — return the most recently created folder.
    # Since we sorted in reverse, the first one is the latest.
    latest = episode_folders[0]
    print(f"  No --episode specified. Using most recent: {latest.name}")
    return latest


def load_episode_files(folder: Path) -> tuple[dict, str, str, dict]:
    """
    Reads the content files from an episode folder.

    Returns a tuple of:
      - research (dict): parsed research_brief.json
      - blog_post (str): contents of blog_post.md
      - podcast_script (str): contents of podcast_script.txt
      - metadata (dict): parsed metadata.json (title + description), or {} if missing

    Exits with a clear error if required files are missing.
    """
    research_path = folder / "research_brief.json"
    blog_path     = folder / "blog_post.md"
    script_path   = folder / "podcast_script.txt"
    metadata_path = folder / "metadata.json"

    for path in [research_path, blog_path, script_path]:
        if not path.exists():
            sys.exit(
                f"[ERROR] Missing file: {path}\n"
                f"        The episode folder may be incomplete. "
                f"Re-run research.py to regenerate."
            )

    # Parse the JSON research brief.
    research = json.loads(research_path.read_text(encoding="utf-8"))

    # Read the blog post and podcast script as plain text.
    blog_post      = blog_path.read_text(encoding="utf-8")
    podcast_script = script_path.read_text(encoding="utf-8")

    # Load metadata if it exists (older episodes may not have it).
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return research, blog_post, podcast_script, metadata


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: NOTION HELPERS
# Convert markdown content into the block format Notion's API expects.
# (These are the same helpers used in research.py — kept here so this script
# works independently without needing to import from research.py.)
# ─────────────────────────────────────────────────────────────────────────────

def _split_text(text: str, max_length: int = 1800) -> list:
    """
    Splits a long string into chunks no longer than max_length characters.
    Tries to split at sentence boundaries ('. ') to keep text readable.

    Why do this? Notion caps each text block at 2000 characters.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while len(text) > max_length:
        # Find the last period before the limit for a clean cut.
        split_at = text.rfind(". ", 0, max_length)
        if split_at == -1:
            split_at = max_length  # No period found — hard cut.
        else:
            split_at += 1  # Keep the period in the first chunk.
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        chunks.append(text)
    return chunks


def _parse_inline_markdown(text: str) -> list:
    """
    Converts inline markdown (bold) into Notion's rich_text format.

    Notion doesn't accept raw markdown. Instead, it uses a list of text objects
    where each object can have annotations like bold: true.

    Example:
      "Hello **world** today" →
      [
        {"type": "text", "text": {"content": "Hello "}},
        {"type": "text", "text": {"content": "world"}, "annotations": {"bold": True}},
        {"type": "text", "text": {"content": " today"}},
      ]
    """
    if not text:
        return [{"type": "text", "text": {"content": ""}}]

    rich_text = []
    # Split on **bold** markers. Odd-indexed parts are the bold ones.
    parts = re.split(r"\*\*(.+?)\*\*", text)

    for idx, part in enumerate(parts):
        if not part:
            continue
        is_bold = (idx % 2 == 1)
        for chunk in _split_text(part, max_length=1800):
            entry = {"type": "text", "text": {"content": chunk}}
            if is_bold:
                entry["annotations"] = {"bold": True}
            rich_text.append(entry)

    return rich_text if rich_text else [{"type": "text", "text": {"content": text}}]


def _markdown_to_notion_blocks(markdown_text: str) -> list:
    """
    Converts a markdown string into a list of Notion block objects.

    Notion's API requires structured JSON blocks — it doesn't render raw markdown.
    This function handles:
      ## Heading    → heading_2 block
      ### Heading   → heading_3 block
      - item        → bulleted_list_item block
      **bold**      → text with bold annotation
      ---           → divider block
      plain text    → paragraph block
    """
    blocks = []
    lines = markdown_text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Skip blank lines.
        if not line:
            i += 1
            continue

        # Horizontal rule → divider block.
        if re.match(r"^-{3,}$", line) or re.match(r"^\*{3,}$", line):
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            i += 1
            continue

        # ## Heading → heading_2 block.
        if line.startswith("## "):
            heading_text = line[3:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": _parse_inline_markdown(heading_text)},
            })
            i += 1
            continue

        # ### Heading → heading_3 block.
        if line.startswith("### "):
            heading_text = line[4:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": _parse_inline_markdown(heading_text)},
            })
            i += 1
            continue

        # Bullet list item (- or *) → bulleted_list_item block.
        if re.match(r"^[\-\*]\s+", line):
            item_text = re.sub(r"^[\-\*]\s+", "", line).strip()
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _parse_inline_markdown(item_text)},
            })
            i += 1
            continue

        # Regular paragraph — collect consecutive non-special lines.
        paragraph_lines = []
        while i < len(lines):
            l = lines[i].rstrip()
            if not l or l.startswith("#") or re.match(r"^[\-\*]\s+", l) or re.match(r"^-{3,}$", l):
                break
            paragraph_lines.append(l)
            i += 1

        if paragraph_lines:
            paragraph_text = " ".join(paragraph_lines)
            # Split long paragraphs into multiple blocks (Notion 2000-char limit).
            for chunk in _split_text(paragraph_text, max_length=1800):
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": _parse_inline_markdown(chunk)},
                })
        continue

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: NOTION INTEGRATION
# Check for existing pages and create the new draft.
# ─────────────────────────────────────────────────────────────────────────────

def check_existing_notion_page(episode_num: int) -> str | None:
    """
    Checks if a Notion page already exists for this episode number.

    Returns the URL of the existing page if found, or None if it doesn't exist.
    This is used to warn the user before creating a duplicate.
    """
    try:
        result = notion.databases.query(
            database_id=NOTION_DB_ID,
            # Filter: only pages where Episode Number equals this episode.
            filter={
                "property": "Episode Number",
                "number": {"equals": episode_num},
            },
        )
        pages = result.get("results", [])
        if pages:
            page = pages[0]
            return page.get("url", f"https://notion.so/{page['id'].replace('-', '')}")
        return None

    except Exception as e:
        print(f"  [Warning] Could not check for existing Notion page: {e}")
        return None


def push_to_notion(
    episode_num: int,
    research: dict,
    blog_post: str,
    podcast_script: str,
    metadata: dict,
) -> str:
    """
    Creates a new Draft page in the Notion database with the episode content.

    This is the same logic as in research.py — the page will have:
    - Title: "Ep. 001 — {short title from metadata}"
    - Status: Draft
    - Episode Number and Publish Date set automatically
    - Full blog post and podcast script in the page body

    Returns:
        The URL of the newly created Notion page.
    """
    # Use the short metadata title if available, otherwise fall back to the episode angle.
    episode_angle = research.get("episode_angle", "Healthcare AI Weekly Roundup")
    short_title = metadata.get("title", episode_angle)
    page_title = f"Ep. {episode_num:03d} — {short_title}"

    print(f"  Creating Notion page: {page_title}")

    # ── Build all the content blocks ─────────────────────────────────────────

    page_blocks = []

    # Callout block at the top showing the episode angle at a glance.
    page_blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": episode_angle}}],
            "icon": {"emoji": "📡"},
            "color": "blue_background",
        },
    })

    # Horizon signals as a quick-reference block at the top of the page.
    horizon    = research.get("horizon", {})
    thirty_day = horizon.get("thirty_days", [])
    sixty_day  = horizon.get("sixty_days", [])

    if thirty_day or sixty_day:
        page_blocks.append({
            "object": "block", "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": "The Horizon"}}]},
        })
        if thirty_day:
            page_blocks.append({
                "object": "block", "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": "30 Days"}}]},
            })
            for signal in thirty_day:
                page_blocks.append({
                    "object": "block", "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": signal}}]},
                })
        if sixty_day:
            page_blocks.append({
                "object": "block", "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": "60 Days"}}]},
            })
            for signal in sixty_day:
                page_blocks.append({
                    "object": "block", "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": signal}}]},
                })

    # Upcoming earnings as a quick reference.
    upcoming = research.get("upcoming_earnings", [])
    if upcoming:
        page_blocks.append({"object": "block", "type": "divider", "divider": {}})
        page_blocks.append({
            "object": "block", "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": "Upcoming Earnings to Watch"}}]},
        })
        for item in upcoming:
            company = item.get("company", "")
            date    = item.get("expected_date", "")
            watch   = item.get("what_to_watch", "")
            text    = f"{company} ({date}) — {watch}"
            page_blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            })

    # Divider, then the full newsletter blog post.
    page_blocks.append({"object": "block", "type": "divider", "divider": {}})
    page_blocks.append({
        "object": "block",
        "type": "heading_1",
        "heading_1": {"rich_text": [{"type": "text", "text": {"content": "Newsletter Post"}}]},
    })
    page_blocks.extend(_markdown_to_notion_blocks(blog_post))

    # Divider, then the full podcast script.
    page_blocks.append({"object": "block", "type": "divider", "divider": {}})
    page_blocks.append({
        "object": "block",
        "type": "heading_1",
        "heading_1": {"rich_text": [{"type": "text", "text": {"content": "Podcast Script"}}]},
    })
    for paragraph in podcast_script.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in _split_text(paragraph, max_length=1800):
            page_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _parse_inline_markdown(chunk)},
            })

    # ── Create the Notion page ────────────────────────────────────────────────

    # Notion only accepts up to 100 blocks in the initial page creation.
    # We send the first 100 now, and append the rest in batches after.
    BATCH_SIZE = 100
    first_batch     = page_blocks[:BATCH_SIZE]
    remaining_blocks = page_blocks[BATCH_SIZE:]

    new_page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "Name":           {"title": [{"type": "text", "text": {"content": page_title}}]},
            "Status":         {"select": {"name": "Draft"}},
            "Episode Number": {"number": episode_num},
            "Publish Date":   {"date": {"start": PUBLISH_DATE.isoformat()}},
        },
        children=first_batch,
    )

    page_id  = new_page["id"]
    page_url = new_page.get("url", f"https://notion.so/{page_id.replace('-', '')}")

    # Append any remaining blocks in batches of 100.
    for i in range(0, len(remaining_blocks), BATCH_SIZE):
        batch = remaining_blocks[i:i + BATCH_SIZE]
        notion.blocks.children.append(block_id=page_id, children=batch)
        print(f"  Appended blocks {i + BATCH_SIZE + 1}–{i + BATCH_SIZE + len(batch)}...")

    return page_url


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main function — finds the episode, reads local files, pushes to Notion.

    Usage:
      python scripts/notion_push.py               # push the most recent episode
      python scripts/notion_push.py --episode 3   # push a specific episode
    """
    parser = argparse.ArgumentParser(
        description="Vitals & Signals — Push local episode files to Notion"
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode number to push (e.g. --episode 3). Defaults to the most recent."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  VITALS & SIGNALS — Notion Push")
    print(f"  Date: {TODAY.isoformat()}")
    print("=" * 60)

    # ── Step 1: Find the episode folder ──────────────────────────────────────
    print("\n[1/3] Finding episode folder...")
    folder = find_episode_folder(args.episode)
    print(f"  Found: {folder}")

    # Extract the episode number from the folder name (e.g. "ep003-..." → 3).
    # We use the folder name rather than the CLI arg because the user might not
    # know the number (e.g. when using the default "most recent" mode).
    folder_name = folder.name  # e.g. "ep003-2024-01-15"
    ep_num_str = folder_name[2:5]   # characters 2-4 are the episode number
    episode_num = int(ep_num_str)
    print(f"  Episode number: {episode_num}")

    # ── Step 2: Load local content files ─────────────────────────────────────
    print("\n[2/3] Loading local content files...")
    research, blog_post, podcast_script, metadata = load_episode_files(folder)
    ftm = research.get("follow_the_money", {})
    print(f"  research_brief.json — {len(research.get('ai_pulse', []))} AI pulse stories, "
          f"{len(ftm.get('startups', []))} funding rounds, "
          f"{len(ftm.get('public_companies', []))} earnings reports")
    print(f"  blog_post.md        — {len(blog_post.split())} words")
    print(f"  podcast_script.txt  — {len(podcast_script.split())} words")
    if metadata.get("title"):
        print(f"  metadata.json       — title: {metadata['title']}")

    # ── Check for existing Notion page ───────────────────────────────────────
    existing_url = check_existing_notion_page(episode_num)
    if existing_url:
        print(f"\n  [Notice] A Notion page for Episode {episode_num} already exists:")
        print(f"  {existing_url}")
        print("  A new Draft page will be created alongside it.")
        print("  (You can manually delete the old one in Notion if needed.)")

    # ── Step 3: Push to Notion ────────────────────────────────────────────────
    print("\n[3/3] Pushing to Notion...")
    page_url = push_to_notion(episode_num, research, blog_post, podcast_script, metadata)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE.")
    print(f"  Episode:      Ep. {episode_num:03d}")
    print(f"  Publish date: {PUBLISH_DATE.isoformat()}")
    print(f"  Notion draft: {page_url}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Open the Notion page and review the content.")
    print("  2. Fill in the editorial feedback properties (Rating, Notes, etc.).")
    print("  3. Change Status from Draft -> Approved when ready.")
    print("  4. Run scripts/publish.py to generate audio and publish.")


if __name__ == "__main__":
    main()
