#!/usr/bin/env python3
"""
Vitals & Signals — Audio Publishing Pipeline
=============================================
This script handles the post-approval publishing flow:

  Step 1 — FIND:    Query Notion for episodes with Status = "Approved"
  Step 2 — READ:    Load the podcast script from the local output/ folder
  Step 3 — AUDIO:   Send the script to ElevenLabs to generate an MP3
  Step 4 — UPLOAD:  Upload MP3 to GitHub Release + update RSS feed.xml
  Step 5 — PUBLISH: Update Notion status to "Published"

Run this script AFTER you've reviewed a Draft episode in Notion and changed
its Status to "Approved". This script will do the rest.

Usage:
  python scripts/publish.py               # finds the most recently approved episode
  python scripts/publish.py --episode 3   # targets a specific episode number
  python scripts/publish.py --no-publish  # generate audio but skip Notion status update
  python scripts/publish.py --no-upload   # skip GitHub upload + RSS update
  python scripts/publish.py --restitch    # re-stitch from cached segments, requires --episode
  python scripts/publish.py --publish-date 2026-04-13  # set a custom RSS pubDate (scheduled release)

Dependencies (all in your venv):
  notion-client, python-dotenv, requests, PyGithub
"""

import os
import sys
import json
import argparse
import datetime
import io
import re
import email.utils
import xml.etree.ElementTree as ET
from pathlib import Path

# python-dotenv lets us load API keys from .env — never hardcode secrets.
from dotenv import load_dotenv

# The official Notion Python SDK.
from notion_client import Client as NotionClient

# requests is a popular library for making HTTP calls (used for ElevenLabs API).
import requests


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# Load all secrets and initialize clients.
# ─────────────────────────────────────────────────────────────────────────────

# Load .env from the project root (one level up from /scripts/).
# Use resolve() for an absolute path and override=True to ensure keys are set
# even when running from a PowerShell session that may have stale env state.
load_dotenv(Path(__file__).parent.parent.resolve() / ".env", override=True)

# Pull all required keys from the environment.
NOTION_API_KEY      = os.getenv("NOTION_API_KEY")
NOTION_DB_ID        = os.getenv("NOTION_DATABASE_ID")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# GitHub credentials — needed for MP3 upload and RSS feed updates.
# Set these in your .env file after creating the GitHub repo.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO  = os.getenv("GITHUB_REPO")   # e.g. "username/vitals-and-signals-feed"

# Validate that all required keys are present before doing anything.
missing = [k for k, v in {
    "NOTION_API_KEY":      NOTION_API_KEY,
    "NOTION_DATABASE_ID":  NOTION_DB_ID,
    "ELEVENLABS_API_KEY":  ELEVENLABS_API_KEY,
    "ELEVENLABS_VOICE_ID": ELEVENLABS_VOICE_ID,
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

# Path to the music folder at the project root.
MUSIC_DIR = Path(__file__).parent.parent / "music"

# Intro music plays at the very start (after the show open).
# The transition swoosh plays between all other sections.
INTRO_MUSIC_PATH       = MUSIC_DIR / "Intro music.mp3"
TRANSITION_SWOOSH_PATH = MUSIC_DIR / "Transition Swoosh.mp3"

# Maps each music marker tag to its corresponding audio file.
MUSIC_MARKER_FILES = {
    "[MUSIC: SHOW OPEN]":          INTRO_MUSIC_PATH,
    "[MUSIC: STORY OPEN]":         TRANSITION_SWOOSH_PATH,
    "[MUSIC: SEGMENT TRANSITION]": TRANSITION_SWOOSH_PATH,
    "[MUSIC: HORIZON TRANSITION]": TRANSITION_SWOOSH_PATH,
    "[MUSIC: OUTRO]":              TRANSITION_SWOOSH_PATH,
}

# If this file exists, it replaces the ElevenLabs-generated show open (seg_01).
# Drop your recorded intro MP3 here — no code changes needed to activate it.
# NOTE: The script text before [MUSIC: SHOW OPEN] is still parsed but ignored for audio
# since your real voice covers the intro. Make sure whatever comes AFTER [MUSIC: SHOW OPEN]
# in the script (the lead hook) is written to flow naturally after your fixed recording.
INTRO_VOICE_FILE = MUSIC_DIR / "intro_voice.mp3"

# Regex that matches any music marker line written into the podcast script.
_MARKER_RE = re.compile(
    r"\[MUSIC: (?:SHOW OPEN|STORY OPEN|SEGMENT TRANSITION|HORIZON TRANSITION|OUTRO)\]"
)


# ElevenLabs API endpoint for text-to-speech.
# We inject the voice ID from .env — this is the voice clone you've set up.
ELEVENLABS_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

# ElevenLabs model to use.
# "eleven_multilingual_v2" is their highest quality model — good for podcasts.
# Switch to "eleven_monolingual_v1" if you want faster/cheaper generation.
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# Today's date — used for filenames and log messages.
TODAY = datetime.date.today()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: NOTION QUERIES
# Find approved episodes and update their status after publishing.
# ─────────────────────────────────────────────────────────────────────────────

def get_approved_episodes() -> list[dict]:
    """
    Queries the Notion database for episodes ready to publish:
    Status = "Approved" (audio not yet generated) or "Generated" (audio done, ready to upload).

    Returns:
        A list of Notion page objects. Each object has the page properties
        (title, episode number, etc.) and the page ID needed to update it.
    """
    try:
        result = notion.databases.query(
            database_id=NOTION_DB_ID,
            # Accept both "Approved" (pre-audio) and "Generated" (audio done, pending upload).
            filter={
                "or": [
                    {"property": "Status", "select": {"equals": "Approved"}},
                    {"property": "Status", "select": {"equals": "Generated"}},
                ]
            },
            # Sort by Episode Number descending so the latest approved is first.
            sorts=[{"property": "Episode Number", "direction": "descending"}],
        )
        return result.get("results", [])

    except Exception as e:
        sys.exit(f"[ERROR] Could not query Notion: {e}")


def extract_episode_number(notion_page: dict) -> int | None:
    """
    Extracts the Episode Number from a Notion page's properties.

    Notion returns properties as nested JSON, so we need to dig into
    the structure to get the actual number value.

    Returns:
        The episode number as an integer, or None if not found.
    """
    ep_prop = notion_page.get("properties", {}).get("Episode Number", {})
    return ep_prop.get("number")


def mark_as_generated(page_id: str) -> None:
    """
    Updates a Notion page's Status to "Generated" after audio is created.
    Requires "Generated" to exist as an option in the Notion Status property.

    Args:
        page_id: The Notion page ID (a long string like "abc123-def456-...").
    """
    notion.pages.update(
        page_id=page_id,
        properties={
            "Status": {"select": {"name": "Generated"}},
        },
    )
    print("  Notion status updated: Approved -> Generated")


def mark_as_published(page_id: str) -> None:
    """
    Updates a Notion page's Status to "Published" after uploading to GitHub and RSS.

    Args:
        page_id: The Notion page ID (a long string like "abc123-def456-...").
    """
    notion.pages.update(
        page_id=page_id,
        properties={
            "Status": {"select": {"name": "Published"}},
        },
    )
    print("  Notion status updated: Generated -> Published")


def update_publish_date_in_notion(page_id: str, publish_date: datetime.date) -> None:
    """
    Updates the "Publish Date" property on a Notion page to the given date.

    Called after audio is generated so the Notion record always reflects the
    intended RSS release date — especially useful when using --publish-date to
    schedule ahead of time.

    Non-fatal: prints a warning if the update fails rather than stopping the run.

    Args:
        page_id:      The Notion page ID.
        publish_date: The intended publication date.
    """
    try:
        notion.pages.update(
            page_id=page_id,
            properties={
                "Publish Date": {"date": {"start": publish_date.isoformat()}},
            },
        )
        print(f"  Notion 'Publish Date' set to {publish_date.isoformat()}")
    except Exception as e:
        print(f"  [WARNING] Could not update Notion Publish Date: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LOCAL FILE LOADING
# Find and read the podcast script from the output/ folder.
# ─────────────────────────────────────────────────────────────────────────────

def find_episode_folder(episode_num: int) -> Path:
    """
    Finds the local output folder for a given episode number.

    Episode folders are named like: ep003-2024-01-15/
    We look for any folder starting with ep{episode_num:03d}-.

    Args:
        episode_num: The episode number (e.g. 3).

    Returns:
        Path to the episode folder.
    """
    if not OUTPUT_DIR.exists():
        sys.exit(
            f"[ERROR] Output directory not found: {OUTPUT_DIR}\n"
            f"        Run scripts/research.py first to generate content."
        )

    prefix = f"ep{episode_num:03d}"
    matches = sorted(OUTPUT_DIR.glob(f"{prefix}*"), reverse=True)  # most recent first

    if not matches:
        sys.exit(
            f"[ERROR] No local folder found for episode {episode_num} "
            f"(looking for {prefix}* in {OUTPUT_DIR}).\n"
            f"        Make sure the output folder exists before running publish.py."
        )

    return matches[0]


def load_podcast_script(folder: Path) -> str:
    """
    Reads the podcast script text from the episode folder.

    Args:
        folder: Path to the episode folder (e.g. output/ep003-2024-01-15/).

    Returns:
        The podcast script as a plain text string.
    """
    script_path = folder / "podcast_script.txt"

    if not script_path.exists():
        sys.exit(
            f"[ERROR] Podcast script not found: {script_path}\n"
            f"        The episode folder may be incomplete."
        )

    script = script_path.read_text(encoding="utf-8").strip()

    if not script:
        sys.exit(f"[ERROR] Podcast script is empty: {script_path}")

    return script


def load_episode_angle(folder: Path) -> str:
    """
    Reads the episode_angle field from research_brief.json.

    This is used as the episode description in the RSS feed item.
    Falls back to a generic description if the file or field is missing.

    Args:
        folder: Path to the episode folder.

    Returns:
        A short description string for the RSS item.
    """
    brief_path = folder / "research_brief.json"
    if brief_path.exists():
        try:
            data = json.loads(brief_path.read_text(encoding="utf-8"))
            angle = data.get("episode_angle", "").strip()
            if angle:
                return angle
        except Exception:
            pass
    return "Weekly intelligence at the intersection of AI and healthcare."


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: ELEVENLABS AUDIO GENERATION
# Send the podcast script to ElevenLabs and get back an MP3.
# ─────────────────────────────────────────────────────────────────────────────

def _split_at_music_markers(script: str) -> list:
    """
    Splits a podcast script at music marker lines and returns a list of
    (text_segment, marker_or_None) tuples.

    The LEAD: metadata line is stripped first (it is not spoken audio).
    Each text_segment is the spoken text before the corresponding marker.
    The final tuple always has marker=None (no music after the last segment).

    Example:
        [  ("show open text", "[MUSIC: SHOW OPEN]"),
           ("hook text",      "[MUSIC: STORY OPEN]"),
           ("lead + roundup", "[MUSIC: HORIZON TRANSITION]"),
           ("horizon",        "[MUSIC: OUTRO]"),
           ("outro",          None)  ]
    """
    # Strip the LEAD: metadata line (it is for reference, not audio).
    script = "\n".join(ln for ln in script.splitlines() if not ln.startswith("LEAD:")).strip()

    # Split the script text into alternating [text, marker, text, marker, ...] parts.
    parts = _MARKER_RE.split(script)
    markers = _MARKER_RE.findall(script)

    # Zip text segments with the marker that follows each one.
    # The last segment has no following marker.
    segments = []
    for i, text in enumerate(parts):
        text = text.strip()
        marker = markers[i] if i < len(markers) else None
        if text or marker:
            segments.append((text, marker))

    return segments


def _split_script(text: str, max_chars: int = 9500) -> list[str]:
    """
    Splits a podcast script into chunks under max_chars, breaking at paragraph boundaries.

    ElevenLabs has a 10,000 character limit per request on the Starter plan.
    We split at double-newline paragraph breaks to keep speech natural at join points.

    Args:
        text: The full podcast script.
        max_chars: Max characters per chunk (default 9500 — leaves headroom below 10k).

    Returns:
        A list of text chunks, each under max_chars.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph would exceed the limit, save the current chunk and start a new one.
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _call_elevenlabs(text: str) -> bytes:
    """
    Sends a single text chunk to ElevenLabs TTS and returns raw MP3 bytes.

    Args:
        text: The text to synthesize (must be under 10,000 characters).

    Returns:
        Raw MP3 audio bytes.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            # Stability (0.0-1.0): higher = more consistent tone, lower = more expressive.
            "stability": 0.5,
            # Similarity boost: how closely to match the original voice clone.
            "similarity_boost": 0.75,
            # Style: adds stylistic variation. 0.0 = neutral.
            "style": 0.0,
            # Speaker boost: clearer, higher quality output.
            "use_speaker_boost": True,
        },
    }

    response = requests.post(
        ELEVENLABS_TTS_URL,
        headers=headers,
        json=payload,
        timeout=300,
        stream=True,
    )

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text[:500]

        if response.status_code == 401:
            sys.exit(
                "[ERROR] ElevenLabs API key is invalid or missing (HTTP 401).\n"
                "        Check ELEVENLABS_API_KEY in your .env file."
            )
        else:
            sys.exit(
                f"[ERROR] ElevenLabs API returned HTTP {response.status_code}.\n"
                f"        Detail: {error_detail}"
            )

    # Collect all streamed bytes into one buffer.
    audio_bytes = b"".join(chunk for chunk in response.iter_content(chunk_size=8192) if chunk)
    return audio_bytes


def generate_audio(podcast_script: str, episode_num: int, output_folder: Path,
                   restitch_only: bool = False) -> Path:
    """
    Generates a complete podcast MP3 in two explicit phases:

    Phase 1 — Voice segments:
        Splits the script at music markers. For each text segment, checks if a
        cached MP3 already exists in output/epXXX-.../segments/. If yes, skips
        ElevenLabs. If no, calls ElevenLabs and saves the result to disk.
        Only proceeds to Phase 2 after ALL segments are confirmed on disk.

    Phase 2 — Stitch:
        Loads each voice segment from disk, normalizes its volume, and interleaves
        the music files. Exports the final combined MP3.

    Args:
        podcast_script:  The full podcast script text.
        episode_num:     Used to name the output file.
        output_folder:   Episode folder — segments/ subfolder is created here.
        restitch_only:   If True, skip Phase 1 entirely (no ElevenLabs calls).
                         All segments must already exist on disk.

    Returns:
        Path to the saved final MP3.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        print("[ERROR] pydub is not installed. Run: pip install pydub")
        print("        Also install ffmpeg: winget install ffmpeg")
        sys.exit(1)

    # Audio settings — tweak these to adjust the sound without re-calling ElevenLabs.
    TARGET_VOICE_DBFS = -18.0   # Target loudness for each voice segment.
    TARGET_MUSIC_DBFS = -28.0   # Music sits 10 dB under voice so it doesn't overpower speech.
    MUSIC_TRIM_MS     = 10_000  # Trim all music clips to first 10 seconds.
    MUSIC_FADE_MS     =  1_000  # Fade music out over the last 1 second.
    SHOW_OPEN_GAP_MS  =  1_000  # Silence before the show open music kicks in.

    # Create the segments subfolder to store individual voice MP3s.
    segments_folder = output_folder / "segments"
    segments_folder.mkdir(exist_ok=True)

    # Split script into (text, marker) pairs at each [MUSIC: ...] line.
    segments = _split_at_music_markers(podcast_script)
    n_text  = sum(1 for text, _ in segments if text)
    n_music = sum(1 for _, m   in segments if m)
    print(f"  {n_text} voice segments + {n_music} music markers found.")
    print(f"  Segments folder: {segments_folder}")

    # ── PHASE 1: Generate voice segments ─────────────────────────────────────
    if restitch_only:
        print("  Phase 1/2: Skipping ElevenLabs (--restitch mode).")
    else:
        print(f"  Phase 1/2: Generating voice segments (Voice: {ELEVENLABS_VOICE_ID}, Model: {ELEVENLABS_MODEL})...")

        # ── Real intro voice: use recorded MP3 instead of ElevenLabs for seg_01 ──
        # If music/intro_voice.mp3 exists, copy it to segments/seg_01_voice.mp3.
        # The ElevenLabs loop below will see seg_01 as "cached" and skip it.
        # This means the show open text in the script is ignored for audio — your
        # real recording plays instead. Everything after [MUSIC: SHOW OPEN] is still
        # generated by ElevenLabs normally.
        intro_dest = segments_folder / "seg_01_voice.mp3"
        if INTRO_VOICE_FILE.exists():
            if not intro_dest.exists():
                import shutil
                shutil.copy2(str(INTRO_VOICE_FILE), str(intro_dest))
                print(f"  Real intro voice copied: {INTRO_VOICE_FILE.name} -> seg_01_voice.mp3")
            else:
                print(f"  Real intro voice: seg_01 already cached — skipping copy.")
        # ─────────────────────────────────────────────────────────────────────────

        for i, (text, marker) in enumerate(segments, start=1):
            if not text:
                continue

            seg_path = segments_folder / f"seg_{i:02d}_voice.mp3"

            if seg_path.exists():
                # Segment already on disk — skip ElevenLabs entirely.
                print(f"  Segment {i}: cached ({seg_path.name})")
            else:
                # Generate via ElevenLabs and save to disk.
                chunks = _split_script(text)
                if len(chunks) > 1:
                    print(f"  Segment {i}: {len(text):,} chars -> {len(chunks)} ElevenLabs chunks...")
                else:
                    print(f"  Segment {i}: {len(text):,} chars -> ElevenLabs...")

                voice_bytes = b""
                for j, chunk in enumerate(chunks, start=1):
                    if len(chunks) > 1:
                        print(f"    Chunk {j}/{len(chunks)}: {len(chunk):,} chars...")
                    voice_bytes += _call_elevenlabs(chunk)

                seg_path.write_bytes(voice_bytes)
                print(f"    Saved: {seg_path.name}")

    # Confirm every expected segment file is present before stitching.
    # Segment 1 is allowed to be absent from disk if there's no text AND no intro file
    # (it's the empty placeholder before [MUSIC: SHOW OPEN] in scripts without a show open).
    missing = [
        f"seg_{i:02d}_voice.mp3"
        for i, (text, _) in enumerate(segments, start=1)
        if text and not (segments_folder / f"seg_{i:02d}_voice.mp3").exists()
    ]
    if missing:
        sys.exit(
            f"[ERROR] Missing voice segments — cannot stitch: {missing}\n"
            f"        Re-run without --restitch to generate them via ElevenLabs."
        )
    print(f"  All {n_text} voice segments confirmed on disk.")

    # ── PHASE 2: Stitch voice + music ────────────────────────────────────────
    print("  Phase 2/2: Stitching voice segments + music...")
    combined = AudioSegment.empty()

    # If a real intro voice file exists, prepend it FIRST — before any music markers.
    # We also mark segment index 1 as "handled" so the main loop below skips it.
    # This works whether the script has show-open text (old episodes) or starts directly
    # at [MUSIC: SHOW OPEN] (new episodes generated after the research.py fix).
    intro_handled = False
    intro_seg_path = segments_folder / "seg_01_voice.mp3"
    if INTRO_VOICE_FILE.exists() and intro_seg_path.exists():
        intro_audio = AudioSegment.from_mp3(str(intro_seg_path))
        if intro_audio.dBFS != float("-inf"):
            gain_needed  = TARGET_VOICE_DBFS - intro_audio.dBFS
            intro_audio  = intro_audio.apply_gain(gain_needed)
        combined     += intro_audio
        intro_handled = True
        print(f"  Added real intro voice ({len(intro_audio)/1000:.1f}s).")

    for i, (text, marker) in enumerate(segments, start=1):

        # Load voice segment from disk and normalize volume.
        # Skip segment 1 if the real intro has already been prepended above.
        if text and not (i == 1 and intro_handled):
            seg_path = segments_folder / f"seg_{i:02d}_voice.mp3"
            voice_audio = AudioSegment.from_mp3(str(seg_path))
            # Guard: skip normalization if the segment is completely silent (dBFS = -inf
            # for a silent clip causes a math error). This also covers intro_voice.mp3.
            if voice_audio.dBFS != float("-inf"):
                gain_needed = TARGET_VOICE_DBFS - voice_audio.dBFS
                voice_audio = voice_audio.apply_gain(gain_needed)
            combined += voice_audio

        # Insert music clip (trimmed + faded).
        if marker:
            music_path = MUSIC_MARKER_FILES.get(marker)
            if music_path and music_path.exists():
                print(f"  Inserting: {music_path.name}")
                music_audio = AudioSegment.from_mp3(str(music_path))
                music_audio = music_audio[:MUSIC_TRIM_MS].fade_out(MUSIC_FADE_MS)
                # Normalize music to TARGET_MUSIC_DBFS so it sits consistently under voice.
                # Order matters: trim -> fade -> normalize (normalize after all processing).
                if music_audio.dBFS != float("-inf"):
                    music_gain  = TARGET_MUSIC_DBFS - music_audio.dBFS
                    music_audio = music_audio.apply_gain(music_gain)
                if marker == "[MUSIC: SHOW OPEN]":
                    combined += AudioSegment.silent(duration=SHOW_OPEN_GAP_MS)
                combined += music_audio
            elif music_path:
                print(f"  [WARNING] Music file not found: {music_path.name} -- skipping {marker}")

    # Export final MP3.
    audio_filename = f"ep{episode_num:03d}-podcast-{TODAY.isoformat()}.mp3"
    audio_path = output_folder / audio_filename
    duration_sec = len(combined) / 1000
    print(f"  Exporting final audio ({duration_sec:.0f}s / {duration_sec/60:.1f} min)...")
    combined.export(str(audio_path), format="mp3")
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"  Audio saved: {audio_path.name} ({file_size_mb:.1f} MB)")

    return audio_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: GITHUB PUBLISHING
# Upload the MP3 to a GitHub Release, then update feed.xml on GitHub Pages.
# ─────────────────────────────────────────────────────────────────────────────

def _get_github_client():
    """
    Returns a PyGithub Github instance, or exits with a clear error if
    PyGithub is not installed or GITHUB_TOKEN / GITHUB_REPO are missing.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        sys.exit(
            "[ERROR] GITHUB_TOKEN and GITHUB_REPO must be set in .env to use upload.\n"
            "        Add them or run with --no-upload to skip GitHub publishing."
        )
    try:
        from github import Github, Auth
        return Github(auth=Auth.Token(GITHUB_TOKEN))
    except ImportError:
        sys.exit(
            "[ERROR] PyGithub is not installed.\n"
            "        Run: pip install PyGithub\n"
            "        Or use --no-upload to skip GitHub publishing."
        )


def upload_to_github_release(audio_path: Path, episode_num: int,
                              title: str, publish_date: datetime.date) -> tuple[str, int]:
    """
    Creates a GitHub Release for the episode and uploads the MP3 as a release asset.

    - Release tag:  ep001, ep002, etc.
    - Release name: Ep. 001 — <title>
    - Asset name:   ep001-podcast-YYYY-MM-DD.mp3

    If a release with the same tag already exists (re-run scenario), the existing
    release is reused and the MP3 asset is deleted + re-uploaded so the URL stays stable.

    Args:
        audio_path:     Local path to the MP3 file.
        episode_num:    Integer episode number (e.g. 1).
        title:          Episode title for the release name.
        publish_date:   Publication date (used in release body).

    Returns:
        (mp3_url, mp3_size_bytes) — stable download URL and file size.
    """
    gh   = _get_github_client()
    repo = gh.get_repo(GITHUB_REPO)
    tag  = f"ep{episode_num:03d}"

    # Check whether a release with this tag already exists.
    existing_release = None
    try:
        existing_release = repo.get_release(tag)
        print(f"  Found existing release: {tag} — will reuse it.")
    except Exception:
        pass  # release doesn't exist yet — we'll create it

    release_name = f"Ep. {episode_num:03d} \u2014 {title}"
    release_body = f"Published: {publish_date.isoformat()}"

    if existing_release is None:
        # Create a new release.
        release = repo.create_git_release(
            tag=tag,
            name=release_name,
            message=release_body,
            draft=False,
            prerelease=False,
        )
        print(f"  Created GitHub release: {release_name}")
    else:
        release = existing_release
        # Delete any existing asset with the same name so we can re-upload cleanly.
        asset_name = audio_path.name
        for asset in release.get_assets():
            if asset.name == asset_name:
                print(f"  Deleting existing asset: {asset_name}")
                asset.delete_asset()
                break

    # Upload the MP3 file as a release asset.
    print(f"  Uploading {audio_path.name} ({audio_path.stat().st_size / (1024*1024):.1f} MB)...")
    asset = release.upload_asset(
        path=str(audio_path),
        content_type="audio/mpeg",
    )

    mp3_url  = asset.browser_download_url
    mp3_size = audio_path.stat().st_size
    print(f"  MP3 uploaded: {mp3_url}")
    return mp3_url, mp3_size


def update_rss_feed(episode_num: int, title: str, description: str,
                    mp3_url: str, mp3_size: int, pub_date: datetime.date) -> None:
    """
    Downloads feed.xml from the GitHub Pages repo, inserts a new <item> at the
    top of the channel (newest first), and pushes it back.

    GitHub Pages redeploys automatically within ~30 seconds of a push.

    Args:
        episode_num:  Integer episode number.
        title:        Episode title (e.g. "Utah Let AI Write Prescriptions...").
        description:  Short episode description from research_brief.json.
        mp3_url:      Stable GitHub Release download URL for the MP3.
        mp3_size:     File size in bytes (required by RSS enclosure element).
        pub_date:     Publication date.
    """
    gh   = _get_github_client()
    repo = gh.get_repo(GITHUB_REPO)

    # Download the current feed.xml content.
    try:
        feed_file    = repo.get_contents("feed.xml")
        feed_xml_raw = feed_file.decoded_content.decode("utf-8")
        feed_sha     = feed_file.sha   # needed for the update API call
    except Exception as e:
        sys.exit(
            f"[ERROR] Could not download feed.xml from {GITHUB_REPO}.\n"
            f"        Make sure feed.xml exists in the repo root.\n"
            f"        Detail: {e}"
        )

    # Parse the XML. We use ElementTree for reading but do a string-based insert
    # so we preserve the existing formatting and namespace declarations.
    try:
        root = ET.fromstring(feed_xml_raw)
    except ET.ParseError as e:
        sys.exit(f"[ERROR] feed.xml is not valid XML: {e}")

    channel = root.find("channel")
    if channel is None:
        sys.exit("[ERROR] feed.xml has no <channel> element.")

    # Format the publication date in RFC 2822 format that podcast directories require.
    # Time is set to 10:00 UTC = 3:00 AM PDT (UTC-7), so scheduled episodes surface
    # early morning Pacific time rather than the previous evening.
    # e.g. "Mon, 13 Apr 2026 10:00:00 +0000"
    pub_date_rfc = email.utils.format_datetime(
        datetime.datetime(pub_date.year, pub_date.month, pub_date.day,
                          hour=10, minute=0, second=0,
                          tzinfo=datetime.timezone.utc)
    )

    # Build the new <item> XML block as a string.
    # We indent it to match the surrounding feed.xml style.
    ep_tag  = f"ep{episode_num:03d}"
    item_xml = f"""
  <item>
    <title>Ep. {episode_num:03d} \u2014 {_xml_escape(title)}</title>
    <description>{_xml_escape(description)}</description>
    <pubDate>{pub_date_rfc}</pubDate>
    <enclosure url="{mp3_url}" length="{mp3_size}" type="audio/mpeg"/>
    <guid isPermaLink="false">vitals-and-signals-{ep_tag}</guid>
    <itunes:episode>{episode_num}</itunes:episode>
    <itunes:episodeType>full</itunes:episodeType>
  </item>"""

    # Check whether this episode is already in the feed (re-run safety).
    guid_marker = f"vitals-and-signals-{ep_tag}"
    if guid_marker in feed_xml_raw:
        print(f"  Episode {ep_tag} already in feed.xml — replacing existing item.")
        # Remove the existing <item> block for this episode.
        # We find the <item> block that contains this guid and remove it.
        feed_xml_raw = _remove_feed_item(feed_xml_raw, guid_marker)

    # Insert the new item just after the last channel-level metadata tag,
    # i.e. right before the first existing <item> or before </channel>.
    first_item_pos = feed_xml_raw.find("<item>")
    close_channel_pos = feed_xml_raw.rfind("</channel>")

    if first_item_pos != -1:
        # Insert before the first existing item (newest-first ordering).
        insert_pos = first_item_pos
    else:
        # No items yet — insert just before </channel>.
        insert_pos = close_channel_pos

    updated_xml = feed_xml_raw[:insert_pos] + item_xml + "\n  " + feed_xml_raw[insert_pos:]

    # Push the updated feed.xml back to GitHub.
    # GitHub Pages will auto-redeploy within ~30 seconds.
    commit_msg = f"Add episode {ep_tag}: {title[:60]}"
    repo.update_file(
        path="feed.xml",
        message=commit_msg,
        content=updated_xml,
        sha=feed_sha,
    )
    print(f"  feed.xml updated and pushed to GitHub.")


def _xml_escape(text: str) -> str:
    """Escapes the five XML special characters in a string."""
    return (text
            .replace("&",  "&amp;")
            .replace("<",  "&lt;")
            .replace(">",  "&gt;")
            .replace('"',  "&quot;")
            .replace("'",  "&apos;"))


def _remove_feed_item(feed_xml: str, guid_marker: str) -> str:
    """
    Removes the <item>...</item> block that contains guid_marker from feed_xml.

    Operates as a simple string search — looks for the <item> tag before
    guid_marker and the </item> tag after it, then removes that block.

    Returns the feed XML string with that item removed.
    """
    guid_pos = feed_xml.find(guid_marker)
    if guid_pos == -1:
        return feed_xml

    # Find the <item> opening tag that comes before the guid.
    item_start = feed_xml.rfind("<item>", 0, guid_pos)
    if item_start == -1:
        return feed_xml

    # Find the </item> closing tag that comes after the guid.
    item_end = feed_xml.find("</item>", guid_pos)
    if item_end == -1:
        return feed_xml
    item_end += len("</item>")

    # Remove the block (and any leading whitespace/newline before it).
    before = feed_xml[:item_start].rstrip(" \t")
    after  = feed_xml[item_end:]
    return before + after


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MAIN ENTRYPOINT
# Orchestrates all steps and prints a final summary.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main function — finds an approved episode, generates audio, uploads to GitHub,
    and marks as published in Notion.

    Usage:
      python scripts/publish.py               # auto-find most recent approved episode
      python scripts/publish.py --episode 3   # target a specific episode
      python scripts/publish.py --no-publish  # skip the Notion status update
      python scripts/publish.py --no-upload   # skip GitHub Release upload + RSS update
      python scripts/publish.py --restitch --episode 3  # re-stitch from cached segments
    """
    parser = argparse.ArgumentParser(
        description="Vitals & Signals — Audio Publishing Pipeline"
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode number to publish (e.g. --episode 3). Defaults to most recently approved."
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="Generate audio but skip updating Notion status to Published. "
             "Useful for testing audio quality before committing."
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip GitHub Release upload and RSS feed update. "
             "Useful for local audio testing without touching the live feed."
    )
    parser.add_argument(
        "--restitch", action="store_true",
        help="Skip ElevenLabs entirely — re-stitch from cached voice segments on disk. "
             "Requires --episode. Use this to adjust music settings without spending quota."
    )
    parser.add_argument(
        "--publish-date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Override the RSS pubDate and Notion Publish Date (e.g. --publish-date 2026-04-13). "
             "Defaults to next Monday. Most podcast apps hold the episode until this date."
    )
    args = parser.parse_args()

    # --restitch requires --episode so we know which folder to load segments from.
    if args.restitch and not args.episode:
        sys.exit("[ERROR] --restitch requires --episode N to identify the episode folder.")

    # ── Resolve publish date ──────────────────────────────────────────────────
    # Default: next Monday from today (same logic as research.py uses for Publish Date).
    days_until_monday = (7 - TODAY.weekday()) % 7 or 7
    publish_date = TODAY + datetime.timedelta(days=days_until_monday)

    if args.publish_date:
        try:
            publish_date = datetime.date.fromisoformat(args.publish_date)
        except ValueError:
            sys.exit(
                f"[ERROR] --publish-date must be YYYY-MM-DD format (e.g. 2026-04-13).\n"
                f"        Got: {args.publish_date!r}"
            )
        if publish_date < TODAY:
            print(f"  [WARNING] --publish-date {publish_date} is in the past. "
                  f"Podcast apps may surface the episode immediately.")
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("  VITALS & SIGNALS — Audio Publishing Pipeline")
    print(f"  Date: {TODAY.isoformat()}")
    print("=" * 60)

    # ── Restitch shortcut ────────────────────────────────────────────────────
    # In --restitch mode, skip Notion entirely — we already know the episode number
    # and just want to re-stitch from cached segments on disk.
    # GitHub upload is also skipped (use --no-upload behaviour implicitly here,
    # since re-stitching implies local audio adjustment, not a publish run).
    if args.restitch:
        print(f"\n[1/5] Restitch mode — skipping Notion lookup.")
        episode_num    = args.episode
        episode_folder = find_episode_folder(episode_num)
        podcast_script = load_podcast_script(episode_folder)
        print(f"\n[2/5] Loaded script from: {episode_folder}")
        print(f"\n[3/5] Re-stitching audio from cached segments...")
        audio_path = generate_audio(podcast_script, episode_num, episode_folder, restitch_only=True)
        print("\n" + "=" * 60)
        print("  DONE (restitch).")
        print(f"  Audio file: {audio_path}")
        print("  GitHub upload and Notion update were skipped.")
        print("=" * 60)
        return

    # ── Step 1: Find the approved episode in Notion ───────────────────────────
    print("\n[1/5] Querying Notion for Approved or Generated episodes...")

    approved_pages = get_approved_episodes()

    if not approved_pages:
        sys.exit(
            "[ERROR] No episodes with Status = 'Approved' or 'Generated' found in Notion.\n"
            "        Go to Notion, open a Draft episode, and change Status to Approved.\n"
            "        Then re-run this script."
        )

    # If --episode was specified, find that specific page.
    if args.episode:
        target_page = None
        for page in approved_pages:
            if extract_episode_number(page) == args.episode:
                target_page = page
                break
        if target_page is None:
            available = [extract_episode_number(p) for p in approved_pages]
            sys.exit(
                f"[ERROR] Episode {args.episode} is not in 'Approved' or 'Generated' status.\n"
                f"        Currently ready episodes: {available}\n"
                f"        Change its Status to Approved in Notion first."
            )
    else:
        # No --episode given — use the most recently approved (highest episode number).
        target_page = approved_pages[0]

    # Extract the episode number and page ID from the Notion page object.
    episode_num = extract_episode_number(target_page)
    page_id     = target_page["id"]

    # Get the page title for display purposes.
    title_prop  = target_page.get("properties", {}).get("Name", {})
    title_parts = title_prop.get("title", [])
    page_title  = title_parts[0]["text"]["content"] if title_parts else f"Ep. {episode_num:03d}"
    # Strip leading "Ep. NNN — " prefix if Notion already includes it,
    # because upload_to_github_release() and update_rss_feed() add it themselves.
    page_title = re.sub(r"^Ep\.\s*\d+\s*[—-]\s*", "", page_title).strip()

    print(f"  Found: {page_title} (Notion ID: {page_id[:8]}...)")

    # ── Step 2: Load the podcast script from local files ──────────────────────
    print(f"\n[2/5] Loading podcast script for Episode {episode_num}...")

    episode_folder = find_episode_folder(episode_num)
    print(f"  Folder: {episode_folder}")

    podcast_script = load_podcast_script(episode_folder)
    word_count = len(podcast_script.split())
    est_minutes = round(word_count / 150)  # ~150 words per minute for natural speech
    print(f"  Script loaded: {word_count} words (~{est_minutes} min)")

    # ── Step 3: Generate audio via ElevenLabs ────────────────────────────────
    print(f"\n[3/5] Generating audio with ElevenLabs...")

    audio_path = generate_audio(podcast_script, episode_num, episode_folder, restitch_only=False)

    # Mark as Generated now that audio is on disk and ready for review.
    print(f"\n  Updating Notion status to Generated...")
    mark_as_generated(page_id)
    # Sync the publish date to Notion so the record matches the scheduled RSS date.
    update_publish_date_in_notion(page_id, publish_date)

    # ── Step 4: Upload to GitHub (Release MP3 + update feed.xml) ─────────────
    mp3_url = None
    if args.no_upload:
        print(f"\n[4/5] Skipping GitHub upload (--no-upload flag set).")
        print(f"  Audio is saved locally only: {audio_path}")
    else:
        print(f"\n[4/5] Uploading to GitHub...")

        # Load the episode description from research_brief.json.
        episode_description = load_episode_angle(episode_folder)

        # publish_date was already resolved above (default: next Monday, or --publish-date override).
        print(f"  Publish date: {publish_date.isoformat()}")

        # Upload MP3 to a GitHub Release.
        mp3_url, mp3_size = upload_to_github_release(
            audio_path   = audio_path,
            episode_num  = episode_num,
            title        = page_title,
            publish_date = publish_date,
        )

        # Update the RSS feed.xml with the new episode item.
        update_rss_feed(
            episode_num = episode_num,
            title       = page_title,
            description = episode_description,
            mp3_url     = mp3_url,
            mp3_size    = mp3_size,
            pub_date    = publish_date,
        )

        print(f"  GitHub publishing complete.")

    # ── Step 5: Update Notion status to "Published" ───────────────────────────
    if args.no_publish:
        print(f"\n[5/5] Skipping final Notion status update (--no-publish flag set).")
        print(f"  Notion page is still marked as 'Generated'.")
    else:
        print(f"\n[5/5] Updating Notion status to Published...")
        mark_as_published(page_id)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE.")
    print(f"  Episode:    {page_title}")
    print(f"  Audio file: {audio_path}")
    print(f"  Publish date: {publish_date.isoformat()}")
    if mp3_url:
        print(f"  MP3 URL:    {mp3_url}")
    if not args.no_publish:
        print(f"  Notion:     Status updated to Published")
    else:
        print(f"  Notion:     Status set to Generated (--no-publish was set, skipped Published)")
    print("=" * 60)
    print("\nNext steps:")
    if args.no_upload:
        print("  1. Listen to the audio file to check quality.")
        print("  2. Run without --no-upload to upload the MP3 and update the RSS feed.")
    else:
        print("  1. Open your RSS feed URL to verify the episode appears.")
        print("     https://github.com/" + (GITHUB_REPO or "your-user/your-repo"))
        print("  2. Paste your feed URL into Spotify for Podcasters or Apple Podcasts Connect.")
    if args.no_publish:
        print("  3. Run without --no-publish to mark the episode as Published in Notion.")


if __name__ == "__main__":
    main()
