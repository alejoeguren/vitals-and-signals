#!/usr/bin/env python3
"""
Vitals & Signals — Research & Writing Pipeline
===============================================
This script is the core of the newsletter/podcast production pipeline.
It runs in three phases:

  Phase 1 — RESEARCH:   A Claude agent searches the web for this week's top
                         AI and healthcare news, organized across four sections.

  Phase 2 — WRITING:    Claude turns the research brief into a full newsletter
                         blog post and a 15-25 minute podcast script.

  Phase 3 — NOTION:     Both pieces are pushed to your Notion database as a
                         new Draft page, ready for your review.

Usage:
  python scripts/research.py
  python scripts/research.py --episode 5        # force a specific episode number
  python scripts/research.py --no-notion        # skip Notion push, just save files

Dependencies (all in your venv):
  anthropic, notion-client, python-dotenv, requests
"""

import os
import re
import sys
import json
import time
import argparse
import datetime
from pathlib import Path

# python-dotenv lets us load API keys from a .env file instead of hardcoding them.
# NEVER put actual keys in your code — always load from environment variables.
from dotenv import load_dotenv

# The official Anthropic Python SDK for calling Claude.
import anthropic

# The official Notion Python SDK for reading/writing to your Notion database.
from notion_client import Client as NotionClient


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# Load all secrets from .env and set up clients + constants.
# ─────────────────────────────────────────────────────────────────────────────

# Load the .env file from the project root (one level up from /scripts/).
# Use resolve() for an absolute path and override=True to ensure keys are set
# even when running from a PowerShell session that may have stale env state.
load_dotenv(Path(__file__).parent.parent.resolve() / ".env", override=True)

# Pull each key from the environment.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NOTION_API_KEY    = os.getenv("NOTION_API_KEY")
NOTION_DB_ID      = os.getenv("NOTION_DATABASE_ID")

# Fail early with a clear message if any key is missing.
missing = [k for k, v in {
    "ANTHROPIC_API_KEY":  ANTHROPIC_API_KEY,
    "NOTION_API_KEY":     NOTION_API_KEY,
    "NOTION_DATABASE_ID": NOTION_DB_ID,
}.items() if not v]
if missing:
    sys.exit(f"[ERROR] Missing required environment variables: {', '.join(missing)}\n"
             f"        Check your .env file in the project root.")

# Initialize API clients.
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
notion = NotionClient(auth=NOTION_API_KEY)

# Model for the research phase — Opus gives the best synthesis across many web
# search results. Switch to "claude-sonnet-4-6" to cut costs if quality holds.
MODEL_RESEARCH = "claude-sonnet-4-6"

# Model for the writing phases (blog post + podcast script). Sonnet is 5x cheaper
# than Opus and produces great output when the facts are already in the brief.
MODEL_WRITING = "claude-sonnet-4-6"

# Max tokens Claude can return in one response.
MAX_TOKENS = 8192

# Where to save generated content locally.
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Date range for research: last 14 days.
# We use 14 days (not 7) to catch earnings calls and slower-moving stories.
TODAY         = datetime.date.today()
TWO_WEEKS_AGO = TODAY - datetime.timedelta(days=14)

# Publish date: next Monday after today.
days_until_monday = (7 - TODAY.weekday()) % 7 or 7
PUBLISH_DATE = TODAY + datetime.timedelta(days=days_until_monday)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PREVIOUS EPISODE DEDUPLICATION
# Load last week's headlines so Claude knows what NOT to repeat.
# ─────────────────────────────────────────────────────────────────────────────

def load_previous_episode_headlines() -> list[str]:
    """
    Reads the most recent episode's research_brief.json and extracts
    every headline that was already covered.

    We inject these into the research prompt so Claude skips repeat stories —
    unless there's been a significant new development (major market reaction,
    new funding, regulatory decision, etc.).

    Returns:
        A list of headline strings, or an empty list if no previous episode exists.
    """
    # Find all episode folders in output/, sorted newest-first.
    folders = sorted(OUTPUT_DIR.glob("ep*"), reverse=True)
    if not folders:
        return []  # First episode — nothing to deduplicate.

    brief_path = folders[0] / "research_brief.json"
    if not brief_path.exists():
        return []

    try:
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
    except Exception:
        return []  # If the file is corrupted, just skip deduplication.

    headlines = []

    # Extract headlines from the current pillar format.
    for story in brief.get("ai_pulse", []):
        if h := story.get("headline"):
            headlines.append(h)

    ftm = brief.get("follow_the_money", {})
    for story in ftm.get("public_companies", []):
        if h := story.get("headline"):
            headlines.append(h)
    for story in ftm.get("startups", []):
        if h := story.get("headline"):
            headlines.append(h)
    for story in ftm.get("acquisitions", []):
        if h := story.get("headline"):
            headlines.append(h)

    for story in brief.get("on_the_radar", []):
        if h := story.get("headline"):
            headlines.append(h)

    # Also handle the old pillar format for backwards compatibility
    # (in case this script runs right after a research.py using the old prompts).
    for story in brief.get("funding_frontier", []):
        if h := story.get("headline"):
            headlines.append(h)
    for story in brief.get("enterprise_signal", []):
        if h := story.get("headline"):
            headlines.append(h)
    for story in brief.get("convergence_thesis", []):
        if h := story.get("headline"):
            headlines.append(h)

    return headlines


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PROMPTS
# The instructions we give Claude for each phase.
# ─────────────────────────────────────────────────────────────────────────────

# --- 3a. Research Agent System Prompt ---
# This is Claude's "character brief" — who it is and how it thinks.
RESEARCH_SYSTEM_PROMPT = """You are a senior analyst for Vitals & Signals, a premium weekly newsletter
covering the intersection of artificial intelligence and healthcare.

Your audience is smart but not all of them are insiders. They include healthcare tech founders,
investors, curious clinicians, and intelligent generalists who follow AI closely.
Write for someone who is sharp and intellectually engaged — but may need one sentence of context
on a niche company or acronym before you explain why it matters. Never condescend. Never over-explain.
Just give enough context to bring everyone into the room before you make your point.

You think both broadly and narrowly:
- Broadly: a new Claude model release matters because of what it unlocks for clinical documentation,
  diagnostics, drug discovery, and hospital operations.
- Narrowly: a small startup like Blueprint.ai deploying AI in mental health is exactly the kind of
  applied story your readers find fascinating.

You connect dots others miss. A new AI chip matters for hospital inference costs.
An FDA ruling on software-as-medical-device matters for every clinical AI startup.
An earnings call from UnitedHealth matters because of what leadership does — and doesn't — say.

Always return factual, specific, sourced information. Prefer concrete details
(dollar amounts, company names, dates, direct quotes) over vague summaries.
Never fabricate facts, funding amounts, or quotes."""


# --- 3b. Research User Prompt (built dynamically to inject previous headlines) ---
def build_research_prompt(previous_headlines: list[str]) -> str:
    """
    Builds the full research prompt, dynamically injecting last week's headlines
    so Claude knows what to skip.

    Args:
        previous_headlines: List of headline strings from the previous episode.
                            Empty list = first episode, no deduplication needed.
    """

    # Build the "already covered" block only if there are previous headlines.
    if previous_headlines:
        covered_lines = "\n".join(f"  - {h}" for h in previous_headlines)
        covered_block = f"""
─────────────────────────────────────────────────────
ALREADY COVERED LAST EPISODE — DO NOT REPEAT
─────────────────────────────────────────────────────
The following stories were covered in the previous episode. Skip them UNLESS there has been
a SIGNIFICANT new development (major market reaction, new funding, regulatory decision,
or major leadership statement that changes the story):

{covered_lines}

"""
    else:
        covered_block = ""

    return f"""Search the web thoroughly for the most important AI and healthcare news
published in the last 14 days (between {TWO_WEEKS_AGO.isoformat()} and {TODAY.isoformat()}).
{covered_block}
Organize your findings across these four sections:

─────────────────────────────────────────────────────
SECTION 1: THE AI PULSE
─────────────────────────────────────────────────────
What's happening in AI broadly — and what does it mean for healthcare?

This section has two layers:

BROAD — Major AI developments from frontier labs and big tech:
  - New model releases (Anthropic, OpenAI, Google DeepMind, Meta, Mistral, xAI)
  - Infrastructure breakthroughs (chips, inference costs, multimodal capabilities)
  - AI safety or regulatory signals from major labs
  For each: always answer "what does this unlock — or threaten — for healthcare?"

NARROW — Specific healthcare AI applications and companies:
  - Healthcare-specific AI deployments (hospitals, clinics, payers, pharma companies)
  - Niche startups applying AI in specific care settings
    (examples: Blueprint.ai in mental health, Abridge in clinical documentation,
     Nabla, Corti, Suki, Ambience — and any others making news this week)
  - EHR integrations, AI diagnostics, ambient documentation, clinical decision support
  - Applied use cases: AI + patient education, AI + prior auth, AI + revenue cycle, etc.

Also capture: What are thought leaders saying?
  - Notable posts, interviews, op-eds, or threads from VCs, CMIOs, researchers, clinicians
  - What debates or tensions are smart people in this space engaging with right now?

Search queries: "healthcare AI {TODAY.year}", "clinical AI deployment", "AI model release healthcare",
"medical AI startup", "AI patient care", "AI healthcare application {TODAY.year}",
site:statnews.com OR site:healthcareitnews.com OR site:modernhealthcare.com OR site:axios.com

─────────────────────────────────────────────────────
SECTION 2: FOLLOW THE MONEY
─────────────────────────────────────────────────────
Where is capital flowing? Cover three sub-sections:

A) PUBLIC COMPANIES — EARNINGS & MARKET SIGNALS
Search for recent earnings calls (last 14 days) from major healthcare and tech companies.
For each relevant earnings report, capture:
  - What leadership specifically said about AI (direct quotes preferred)
  - AI-related risk factors disclosed in the report
  - AI-driven revenue gains or cost savings reported
  - How the market reacted (stock movement %, analyst commentary)

Key companies to check: UnitedHealth/Optum, CVS/Aetna, Humana, Elevance Health,
Epic Systems, Oracle Health, Veeva, Teladoc, Doximity, Alphabet/Google Health,
Microsoft (Nuance/Copilot), Amazon (AWS Health), Apple, Pfizer, Eli Lilly,
AstraZeneca, Roche, HCA Healthcare, Tenet Health, CommonSpirit, Accenture Health.

ALSO: Identify any UPCOMING earnings calls in the next 30 days from these companies.
For each, flag what AI-related topics listeners should watch for.

Search queries: "earnings call AI {TODAY.year}", "UnitedHealth earnings AI",
"healthcare earnings AI technology", "{TODAY.year} earnings transcript healthcare AI",
site:seekingalpha.com OR site:fool.com OR site:bloomberg.com OR site:wsj.com

B) STARTUPS — FUNDING ROUNDS
  - Company name, funding amount, round stage (Seed / Series A / B / C), lead investors
  - What the company does in plain language (one sentence — assume the reader hasn't heard of them)
  - What this funding likely enables — what's the probable strategic play?

Search queries: "healthcare AI funding {TODAY.year}", "digital health series A B C",
"health AI venture capital", site:techcrunch.com OR site:axios.com OR site:fiercehealthcare.com

C) STRATEGIC ACQUISITIONS
  - Who acquired whom, and for how much (if disclosed)
  - What AI capability does the acquirer gain?
  - What does this signal about their broader AI strategy?

Search queries: "healthcare AI acquisition {TODAY.year}", "health tech merger acquisition",
"digital health acquisition", site:axios.com OR site:reuters.com OR site:bloomberg.com

─────────────────────────────────────────────────────
SECTION 3: ON THE RADAR
─────────────────────────────────────────────────────
A catch-all for anything important that doesn't fit the sections above:
  - FDA clearances or rejections of AI-enabled medical devices
  - Published research from top journals (NEJM, JAMA, Nature Medicine, The Lancet)
  - Policy and regulatory signals (CMS, ONC, HHS, FTC, EU AI Act, international regulators)
  - Clinical trial results involving AI tools
  - Patient safety concerns or adverse events involving AI
  - International healthcare AI news worth tracking

Search queries: "FDA AI clearance {TODAY.year}", "healthcare AI regulation {TODAY.year}",
"AI clinical trial results", "medical AI research paper",
site:nejm.org OR site:jamanetwork.com OR site:nature.com OR site:thelancet.com

─────────────────────────────────────────────────────

After completing your research, return a structured JSON object — and ONLY the JSON object,
no markdown code fences, no preamble, no explanation. Use exactly this structure:

{{
  "week_of": "{TODAY.isoformat()}",
  "ai_pulse": [
    {{
      "headline": "One-line headline",
      "type": "broad or narrow",
      "summary": "2-3 sentence summary with specific details",
      "key_facts": ["Specific fact 1", "Specific fact 2", "Specific fact 3"],
      "healthcare_angle": "What this means specifically for healthcare — one sentence",
      "source": "Publication name or URL"
    }}
  ],
  "follow_the_money": {{
    "public_companies": [
      {{
        "headline": "One-line headline",
        "company": "Company name",
        "summary": "What happened in the earnings report",
        "ai_commentary": "What leadership said about AI (direct quote if available)",
        "risk_factors": ["AI-related risk factor disclosed"],
        "market_reaction": "Stock movement and analyst commentary",
        "source": "Publication name or URL"
      }}
    ],
    "startups": [
      {{
        "headline": "One-line headline",
        "company": "Company name",
        "amount": "Funding amount (e.g. $45M)",
        "stage": "Seed / Series A / Series B / etc.",
        "investors": ["Lead investor", "Other investor"],
        "what_they_do": "Plain-language description — one sentence",
        "likely_strategy": "What this funding probably enables",
        "source": "Publication name or URL"
      }}
    ],
    "acquisitions": [
      {{
        "headline": "One-line headline",
        "acquirer": "Acquiring company",
        "target": "Acquired company",
        "amount": "Deal value if disclosed, otherwise 'undisclosed'",
        "ai_capability_gained": "What AI capability this adds to the acquirer",
        "strategic_signal": "What this tells us about the acquirer's AI strategy",
        "source": "Publication name or URL"
      }}
    ]
  }},
  "on_the_radar": [
    {{
      "headline": "One-line headline",
      "category": "FDA / Research / Policy / Clinical / International / Other",
      "summary": "2-3 sentence summary with specific details",
      "why_it_matters": "One sentence on significance for the Vitals & Signals audience",
      "source": "Publication name or URL"
    }}
  ],
  "upcoming_earnings": [
    {{
      "company": "Company name",
      "expected_date": "Approximate date or 'week of YYYY-MM-DD'",
      "what_to_watch": "Specific AI-related topic or question to watch for"
    }}
  ],
  "horizon": {{
    "thirty_days": [
      "Specific, named signal to watch in the next 30 days"
    ],
    "sixty_days": [
      "Structural shift to watch over the next 60 days"
    ]
  }},
  "episode_angle": "The single most compelling narrative angle for this week — one punchy sentence"
}}

Aim for: 3-5 items in ai_pulse, 2-4 public company earnings, 3-5 startup rounds,
1-3 acquisitions, 3-5 on_the_radar items, 3-5 upcoming earnings,
3-4 thirty-day signals, 2-3 sixty-day signals.
Be specific. Be factual. Return only valid JSON."""


def build_scout_prompt(previous_headlines: list[str]) -> str:
    """
    Builds the scout phase prompt. Asks Claude to do 3 broad searches and return
    a lightweight JSON summary of the top stories + a recommended lead.
    """
    if previous_headlines:
        covered_lines = "\n".join(f"  - {h}" for h in previous_headlines)
        covered_block = f"""
Already covered last episode — skip these unless there's a major new development:
{covered_lines}
"""
    else:
        covered_block = ""

    return f"""You are doing a quick headline sweep for the Vitals & Signals newsletter.
Search the web for the most important AI and healthcare news from the last 14 days
(between {TWO_WEEKS_AGO.isoformat()} and {TODAY.isoformat()}).
{covered_block}
Do exactly 3 broad searches to get a wide view of the week. Good starting queries:
- "healthcare AI news {TODAY.year} {TODAY.strftime('%B')}"
- "health AI funding acquisition earnings {TODAY.year}"
- "FDA AI HIMSS clinical AI {TODAY.year}"

Then return ONLY a JSON object — no preamble, no markdown fences — with this structure:
{{
  "top_stories": [
    {{
      "headline": "One-line headline",
      "category": "ai_pulse | funding | earnings | regulatory | other",
      "why_compelling": "One sentence on why this matters for healthcare AI founders/investors/clinicians",
      "approximate_date": "YYYY-MM-DD or 'week of YYYY-MM-DD'"
    }}
  ],
  "recommended_lead": "The headline of the single best story for an 8-10 minute podcast deep dive",
  "lead_rationale": "2-3 sentences: why this story has the most narrative depth, human stakes, and relevance this week"
}}

Aim for 8-12 top stories. Pick the lead story based on: newsworthiness, narrative depth,
human stakes, and how much there is to actually dig into. A story that just announced a
funding round is weaker than a story with a controversy, pivot, or surprising implication."""


def build_deep_research_prompt(scout_results: dict) -> str:
    """
    Builds the deep research prompt. Injects the scout results so Claude knows
    the lead story, then asks for a deep dive + roundup sweep.
    """
    lead = scout_results.get("recommended_lead", "the top story from your research")
    rationale = scout_results.get("lead_rationale", "")

    # Format the other top stories as context for the roundup
    top_stories = scout_results.get("top_stories", [])
    other_stories = [s for s in top_stories if s.get("headline") != lead]
    other_lines = "\n".join(
        f"  - [{s.get('category','other')}] {s.get('headline','')}"
        for s in other_stories[:10]
    )

    return f"""You are now doing the deep research phase for this week's Vitals & Signals episode.

Your scout research identified these top stories this week:
{other_lines}

YOUR LEAD STORY FOR THIS EPISODE:
  "{lead}"

WHY THIS IS THE LEAD:
  {rationale}

Now do two things using up to 8 searches:

1. DEEP DIVE on the lead story (use 4-5 searches):
   Search for everything you need to write an 8-10 minute podcast segment on this story.
   You need: origin/background, exactly what happened, key players and their quotes,
   what it means for healthcare AI broadly, human stakes, expert reactions,
   and what this signals for the industry in the next 30-60 days.

2. ROUNDUP SWEEP (use 3-4 searches):
   Quick sweeps to fill out the other sections. Cover:
   - AI Pulse: other notable AI/healthcare deployments, model releases, thought leadership
   - Follow the Money: earnings calls with AI commentary, startup funding rounds, acquisitions
   - On the Radar: FDA clearances, research papers, regulatory signals, policy updates

After your searches, return the full research brief as a JSON object — ONLY the JSON,
no markdown fences, no preamble. Use this exact structure:

{{
  "week_of": "{TODAY.isoformat()}",
  "ai_pulse": [
    {{
      "headline": "One-line headline",
      "type": "broad or narrow",
      "summary": "2-3 sentence summary with specific details",
      "key_facts": ["Specific fact 1", "Specific fact 2", "Specific fact 3"],
      "healthcare_angle": "What this means specifically for healthcare",
      "source": "Publication name or URL"
    }}
  ],
  "follow_the_money": {{
    "public_companies": [
      {{
        "headline": "One-line headline",
        "company": "Company name",
        "summary": "What happened",
        "ai_commentary": "What leadership said about AI (direct quote if available)",
        "risk_factors": ["AI-related risk factor"],
        "market_reaction": "Stock movement and analyst commentary",
        "source": "Publication name or URL"
      }}
    ],
    "startups": [
      {{
        "headline": "One-line headline",
        "company": "Company name",
        "amount": "Funding amount",
        "stage": "Seed / Series A / B / C / etc.",
        "investors": ["Lead investor"],
        "what_they_do": "Plain-language description",
        "likely_strategy": "What this funding probably enables",
        "source": "Publication name or URL"
      }}
    ],
    "acquisitions": [
      {{
        "headline": "One-line headline",
        "acquirer": "Acquiring company",
        "target": "Acquired company",
        "amount": "Deal value or 'undisclosed'",
        "ai_capability_gained": "What AI capability this adds",
        "strategic_signal": "What this tells us about the acquirer's AI strategy",
        "source": "Publication name or URL"
      }}
    ]
  }},
  "on_the_radar": [
    {{
      "headline": "One-line headline",
      "category": "FDA / Research / Policy / Clinical / International / Other",
      "summary": "2-3 sentence summary",
      "why_it_matters": "One sentence on significance",
      "source": "Publication name or URL"
    }}
  ],
  "upcoming_earnings": [
    {{
      "company": "Company name",
      "expected_date": "Approximate date",
      "what_to_watch": "Specific AI-related topic to watch"
    }}
  ],
  "horizon": {{
    "thirty_days": ["Specific signal to watch in the next 30 days"],
    "sixty_days": ["Structural shift to watch over the next 60 days"]
  }},
  "episode_angle": "The single most compelling narrative angle for this week — one punchy sentence"
}}

The lead story should be reflected in the episode_angle. Aim for 3-5 ai_pulse items,
2-4 public company earnings, 3-5 startups, 1-3 acquisitions, 3-5 on_the_radar items.
Be specific and factual. Return only valid JSON."""


# --- 3c. Compact Research Helper ---
def _compact_research(research: dict) -> dict:
    """
    Returns a slimmed-down version of the research brief for use in writing prompts.

    The full research brief contains key_facts arrays, source URLs, investor lists,
    and other fields that are useful for fact-checking but add token cost when passed
    to the blog/podcast writing prompts. The writer only needs the narrative content
    (headlines, summaries, angles, and strategic signals).

    Removing these fields cuts writing prompt input tokens by ~40%, saving ~$0.15/run.
    The full brief is still saved to disk so nothing is permanently lost.
    """
    ftm = research.get("follow_the_money", {})
    return {
        "week_of":       research.get("week_of"),
        "episode_angle": research.get("episode_angle"),
        "ai_pulse": [
            {
                "headline":         s.get("headline"),
                "type":             s.get("type"),
                "summary":          s.get("summary"),
                "healthcare_angle": s.get("healthcare_angle"),
            }
            for s in research.get("ai_pulse", [])
        ],
        "follow_the_money": {
            "public_companies": [
                {
                    "headline":       s.get("headline"),
                    "company":        s.get("company"),
                    "summary":        s.get("summary"),
                    "ai_commentary":  s.get("ai_commentary"),
                    "market_reaction": s.get("market_reaction"),
                }
                for s in ftm.get("public_companies", [])
            ],
            "startups": [
                {
                    "headline":        s.get("headline"),
                    "company":         s.get("company"),
                    "amount":          s.get("amount"),
                    "stage":           s.get("stage"),
                    "what_they_do":    s.get("what_they_do"),
                    "likely_strategy": s.get("likely_strategy"),
                }
                for s in ftm.get("startups", [])
            ],
            "acquisitions": [
                {
                    "headline":             s.get("headline"),
                    "acquirer":             s.get("acquirer"),
                    "target":               s.get("target"),
                    "amount":               s.get("amount"),
                    "ai_capability_gained": s.get("ai_capability_gained"),
                    "strategic_signal":     s.get("strategic_signal"),
                }
                for s in ftm.get("acquisitions", [])
            ],
        },
        "on_the_radar": [
            {
                "headline":       s.get("headline"),
                "category":       s.get("category"),
                "summary":        s.get("summary"),
                "why_it_matters": s.get("why_it_matters"),
            }
            for s in research.get("on_the_radar", [])
        ],
        "upcoming_earnings": research.get("upcoming_earnings", []),
        "horizon":           research.get("horizon", {}),
    }


# --- 3c. Blog Post Prompt ---
BLOG_SYSTEM_PROMPT = """You are the writer and editor of Vitals & Signals.

Your audience: healthcare tech founders, investors, clinicians, and curious generalists
who follow AI closely. They are smart and engaged — but not all of them are deep insiders.
Write as if you're talking to a brilliant friend who is intellectually curious but may need
one sentence of context on a niche company or technical concept before you explain why it matters.
Never condescend. Never over-explain. Just bring everyone into the room, then make your point.

Your voice: Sharp, direct, forward-looking. You have opinions and you name them.
Not academic. Not breathless. Authoritative without being stiff.
Think: Stratechery, Not Boring, The Diff — applied to healthcare AI.
Data-driven prose, not bullet-point reports."""

def build_blog_prompt(research: dict) -> str:
    """
    Builds the writing prompt for the blog post phase.
    We pass a compact version of the research brief to reduce token cost — it has
    everything the writer needs (headlines, summaries, angles) without the verbose
    fields (key_facts arrays, source URLs) that are only useful for fact-checking.
    """
    return f"""Write a full issue of the Vitals & Signals newsletter based on this week's research brief.

RESEARCH BRIEF:
{json.dumps(_compact_research(research), indent=2)}

─────────────────────────────────────────────────────
STRUCTURE — use these exact section headers:
─────────────────────────────────────────────────────

## Opening Hook
2-3 strong paragraphs. Open with a striking observation, stat, or tension — something that
makes the reader stop scrolling. Zoom out to the week's overarching theme.
End with one sentence previewing what's inside.

## The AI Pulse
4-6 paragraphs. Cover the week's most important AI developments — both broad
(what frontier labs and big tech released or announced) and narrow (specific healthcare
deployments, applied use cases, niche startups).
For every broad AI story, always answer: what does this mean for healthcare?
Briefly introduce any companies readers may not know before explaining why they matter.
Weave in thought leader perspectives where they add texture.

## Follow the Money
4-6 paragraphs. Don't silo earnings, startups, and acquisitions — let them flow together
as a connected narrative about where capital is moving and why.
What do the earnings signals reveal about how incumbents are betting on AI?
What do the startup rounds tell us about where the smart money sees opportunity?
What do the acquisitions signal about strategic priorities?
Include specific numbers: funding amounts, stock movements, deal values.
Flag any upcoming earnings calls worth watching.

## On the Radar
3-4 tight paragraphs covering the catch-all stories.
FDA clearances, research papers, policy moves — keep it punchy.
This should feel like the "before you go" segment of a great briefing.

## The Horizon
A forward-looking section split into two clear parts:

**30 Days — Signals to Watch**
3-4 bullets. Specific and named — not "watch AI regulation" but
"Watch for CMS's final rule on AI-assisted prior auth, expected by [date]."
Format: "**Company or Topic** — what to watch and why."

**60 Days — Structural Shifts**
2-3 bullets. Zoom out. What larger pattern is beginning to take shape?
Format: "**Theme** — what the signal is and what it could mean."

## Closing
1-2 paragraphs. Bring the week's theme full circle.
Leave the reader with one clear takeaway or mental model. No fluff.

─────────────────────────────────────────────────────
STYLE RULES:
─────────────────────────────────────────────────────
- Briefly introduce niche companies before explaining why they matter
  ("Blueprint.ai, a mental health AI startup, announced..." not just "Blueprint.ai announced...")
- Define acronyms on first use (EHR, CMS, ONC, FDA, prior auth, etc.)
- Use specific numbers and company names — never "a large health system" when you know the name
- Use second person occasionally ("your portfolio", "your hospital") to pull the reader in
- Every sentence earns its place. Cut hedging language.
- Target length: 1,500–2,000 words
- Clean markdown: ## for H2 sections, **bold** for key terms and company names on first mention
- Do NOT include a title, byline, or date — those are added separately

Write the complete newsletter issue now."""


# --- 3d. Podcast Script Prompt ---
PODCAST_SYSTEM_PROMPT = """You are writing a solo podcast script for Vitals & Signals.

The host's name is Aahlay (spelled Ale). Use this spelling throughout — it is optimized
for ElevenLabs text-to-speech pronunciation. Do NOT write a show introduction — the
fixed show open is handled separately. Start directly with the episode hook.

FORMAT: Lead + Roundup
This is not a news briefing. It is a narrative podcast with a roundup section.

PART 1 — THE LEAD (~8-10 minutes / ~1,200-1,500 words)
One story, told as a proper narrative. It has five beats:
  Beat 1 — The Setup: What was the world like before this week's development?
  Beat 2 — The Event: What happened, in sequence, with specific details?
  Beat 3 — The Pivot: Who saw this coming and what did they do about it?
  Beat 4 — The Landscape: Who wins, who loses, what's the honest analysis?
  Beat 5 — The Human Stakes: What does this mean beyond the business story?

PART 2 — THE ROUNDUP (~3-4 minutes / ~450-600 words)
All other important stories from the week. 2–3 sentences per item. Fast and telegraphed.
Open with: "Quick hits — things you need before your next board meeting or your next rounds."
Group loosely by theme (models/infrastructure, money, regulatory) but keep it flowing prose,
not headers.

PART 3 — THE HORIZON (~2-3 minutes / ~250-350 words)
30-day signals: specific, named, with expected dates and outcomes. 3 items.
60-day structural call: one bold thesis about what's shifting. 1–2 items.

OUTRO (~1 minute / ~100-150 words)
Close the loop from the hook. Reframe the opening tension as resolved or recast.
End on one thing the listener carries into their week.

---

VOICE AND STYLE

The host speaks like a sharp, well-informed friend explaining something important over coffee.
Not a news anchor. Not a professor. Someone who has done the reading so you don't have to,
and has strong opinions about what it all means.

Core voice principles:
- First person throughout ("I think", "My read on this is", "Here's what I find interesting")
- "Look," and "Here's the thing:" to signal key moments
- Contractions always ("it's", "we're", "they've") — never formal prose
- Short paragraphs: 2–4 sentences maximum
- No bullet points, no numbered lists, no section headers in the output
- Verbal transitions between sections — not written headers

The script will be processed by ElevenLabs text-to-speech. Write for the ear, not the eye.

---

SENTENCE RHYTHM — THIS IS THE MOST IMPORTANT TECHNICAL REQUIREMENT

ElevenLabs delivers natural speech when the script has rhythm variation.
Monotone delivery happens when sentences are all the same length.

RULE: Never write more than 3 sentences of similar length in a row.

Use three sentence types deliberately:
- SHORT (under 8 words): Key takeaways, reversals, dramatic beats. Forces a pause.
- MEDIUM (10–18 words): Conversational baseline. Building context.
- LONG (20+ words): Analysis and setup before a payoff.

Example of weak rhythm (avoid):
  "UnitedHealth reported Q4 revenue of $113 billion, which missed estimates by $530 million.
  The stock dropped 16 percent in pre-market trading. Leadership announced plans for $1 billion
  in AI cost reductions for 2026."

Example of strong rhythm (use):
  "UnitedHealth missed Q4 estimates by $530 million and guided for its first revenue decline
  in a decade. The stock dropped 16 percent before markets opened. That's the headline.
  But here's what I actually want you to notice."

Short sentence. Punchy landing. Then pivot. That's the pattern.

---

NARRATIVE TECHNIQUES — APPLY ALL OF THESE

1. SIGNIFICANCE BEFORE EVIDENCE
   Lead with the "so what" before the "what". Never introduce a company name or funding
   amount before the listener knows why it matters.

   Weak: "Synthpop raised a $15M Series A led by Ansa Capital..."
   Strong: "A former Humana CEO just wrote a check into a company that automates prior auth.
            His name is Bruce Broussard. The company is Synthpop."

2. RHETORICAL QUESTIONS (minimum 4–5 in the lead story alone)
   Use them to advance the narrative, not just mark transitions.
   "How did this happen so fast?" / "So where does that leave everyone else?" /
   "The question is: what do the startups do next?"

3. ANALOGY (at least one per episode)
   Make unfamiliar concepts familiar with one real-world comparison.
   Example: "It happened to calendar apps when smartphones launched. It happened to
   standalone navigation apps when Apple and Google built maps into the operating system.
   And now it's happening to ambient scribing."

4. PROOF TYPE VARIETY
   Use at least 4 of these 7 types: statistics, dollar amounts, direct quotes,
   analogies, contrasts, timelines, personal reads ("My read is...").

5. CALLBACKS
   Reference the hook at least once mid-episode and again in the outro.
   "Remember what I said at the top about how a category dies? Here's what that looks like
   from the inside."

6. THE HUMAN STAKES BEAT
   Every lead story needs one moment that goes beyond the business story.
   What does this mean for physicians? Patients? Clinicians doing their jobs at midnight?
   This beat is what separates analysis from journalism.
   Do NOT save this for the end. Weave a human moment in earlier — then return to it.

7. CONCRETE SCENES — SHOW, DON'T BRIEF
   Ground every major claim in a specific person, place, or moment.
   Not "a patient with diabetes" — "a sixty-two-year-old woman in Memphis who sees her
   cardiologist twice a year and gets a call from nobody in between."
   Not "AI companies are positioning for this" — "Pair Team, a Kleiner Perkins-backed startup
   that spent three years doing the unglamorous work of medication adherence calls..."
   At least two of the five beats should open with or anchor in a specific scene.
   The listener should be able to see the story, not just understand it.

---

BANNED PATTERNS — DO NOT USE THESE

These structures are overused AI writing patterns. They flatten the voice and make the
script sound templated. Avoid them entirely or use each AT MOST ONCE per episode.

1. TRIPLE NEGATION REVEAL
   "Not a funding round. Not a model release. Not an acquisition. A payment rail."
   Powerful once, exhausting twice. If you use it at all, use it only at the most climactic
   single moment of the episode — then find other ways to land reveals:
   - Build through narrative: "The industry expected X. What they got was Y."
   - Cut straight to the punch: "The actual number is sixteen billion. Not million. Billion."
   - Use contrast: "Everyone was watching the labs. Nobody was watching CMS."

2. REPETITIVE PARAGRAPH OPENERS
   Do not start more than two paragraphs in a single section with "Now" or "Here's".
   Vary openers: use specific dates, names, numbers, or scene-setting to open paragraphs.
   Weak: "Now here's the thing. Now here's what changed. Now here's the number."
   Strong: "July fifth is when it goes live." / "Pair Team saw this coming." / "Sixteen weeks."

3. TELEGRAPHED EMPHASIS
   "Read that sentence again." / "Sit with that." / "Let that land."
   These are crutches. If the sentence is good enough, it lands without the stage direction.
   Earn the emphasis through rhythm and placement, not instruction.

4. CONSECUTIVE IDENTICAL SENTENCE STARTS
   Never start three or more sentences in a row with the same word.
   Especially watch for: "Not...", "The...", "And...", "It's..."

---

ELEVENLABS FORMATTING RULES

DO:
- Use em dashes — for mid-thought interruptions and beat pauses
- Use <break time="1.5s" /> at major narrative beat transitions (2–3 times in the lead, max)
- Write numbers as spoken words: "113 billion dollars", "February fifth", "sixteen percent"
- Use paragraph breaks generously — ElevenLabs respects spacing as breathing room
- Use ellipsis ... sparingly for suspense or trailing thought (adds hesitation quality)

DO NOT:
- Use [emotional direction] brackets — ElevenLabs reads them literally
- Use [pause] or [beat] — spoken aloud
- Write large numbers as numerals: not "$113,200,000,000"
- Write more than 4 sentences in a paragraph
- Use headers or bullet points anywhere in the script output

MUSIC MARKERS (include these exactly — they are stripped before ElevenLabs processing):
  [MUSIC: SHOW OPEN]         — after the fixed show open, before the hook
  [MUSIC: STORY OPEN]        — after the hook, before Beat 1 of the lead
  [MUSIC: SEGMENT TRANSITION] — between the lead and the roundup
  [MUSIC: HORIZON TRANSITION] — before The Horizon section
  [MUSIC: OUTRO]             — before the closing paragraph

---

OUTPUT INSTRUCTION

Output only the script — no preamble, no word count, no meta-commentary.
Do not include the fixed show open (it is prepended separately in production).
Start with LEAD: [story name] on the first line, then [MUSIC: SHOW OPEN], then the hook.

"""

def build_podcast_prompt(research: dict) -> str:
    """
    Builds the writing prompt for the podcast script.

    Format: Lead + Roundup (see PODCAST_SYSTEM_PROMPT for full format spec).

    We give Claude the compact research brief for factual grounding.
    The prompt instructs Claude to identify the lead story before writing,
    so the selection is deliberate and documented in the output.
    """
    compact = _compact_research(research)
    episode_angle = research.get("episode_angle", "")

    return f"""Write a complete podcast script for Vitals & Signals using the Lead + Roundup format.

EPISODE ANGLE (use this as your north star for picking the lead story):
{episode_angle}

RESEARCH BRIEF (factual grounding — use specific details, names, numbers):
{json.dumps(compact, indent=2)}

---

STEP 1 — PICK YOUR LEAD STORY

Before writing, identify which single story from the research brief has:
  - A clear narrative arc (setup, event, turning point, resolution)
  - The most direct connection to multiple other stories this week
  - The strongest "so what" for a healthcare founder, investor, or clinician

State your lead story selection in ONE line at the very top of your output, like this:
  LEAD: [story name]

Then place [MUSIC: SHOW OPEN] immediately after the LEAD line, followed by the hook.

---

STEP 2 — WRITE THE SCRIPT

Follow the Lead + Roundup format exactly as specified in the system prompt.

Checklist before you output:
  [ ] Hook creates genuine tension or contradiction in the first 60 seconds
  [ ] Lead story has all five beats (setup, event, pivot, landscape, human stakes)
  [ ] At least one analogy used to make an unfamiliar concept familiar
  [ ] At least 4 rhetorical questions in the lead story
  [ ] Significance stated before evidence for every major story
  [ ] Sentence rhythm varies — no more than 3 same-length sentences in a row
  [ ] At least 2 <break time="1.5s" /> tags at major narrative beat transitions
  [ ] All 5 music markers present: SHOW OPEN, STORY OPEN, SEGMENT TRANSITION,
      HORIZON TRANSITION, OUTRO
  [ ] Numbers written as spoken words throughout
  [ ] Outro closes the loop from the hook

Target length: 2,000-2,500 words (~13-17 minutes at 150 wpm).
Hard maximum: 3,000 words / 20 minutes. Aim for ~15 minutes.
Output only the script with the LEAD line at top. No other meta-commentary."""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RESEARCH AGENT
# Two-phase scout-then-dig approach:
#   Phase A (Scout):       3 broad searches → lightweight JSON of top stories + recommended lead
#   Phase B (Deep Research): 8 focused searches → full research brief JSON
# ─────────────────────────────────────────────────────────────────────────────

def _run_agentic_loop(prompt: str, tools: list, max_tokens: int, label: str) -> object:
    """
    Shared agentic loop used by both scout and deep research phases.
    Runs API calls until Claude signals end_turn, handling tool_use turns in between.

    Args:
        prompt:     The user message to start the conversation.
        tools:      The tools list (web_search with max_uses set).
        max_tokens: Max output tokens for this phase.
        label:      Short label for print statements ("Scout" or "Deep").

    Returns:
        The final Claude response object (stop_reason == "end_turn").
    """
    messages = [{"role": "user", "content": prompt}]

    for iteration in range(1, 25):
        response = _claude_create_with_retry(
            model=MODEL_RESEARCH,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": RESEARCH_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=tools,
            messages=messages,
        )

        # Serialize content blocks to dicts and strip trailing whitespace from text blocks.
        # The API returns a 400 if the last assistant text block has trailing whitespace
        # when followed by a tool_use turn — this sanitization prevents that error.
        sanitized_content = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                sanitized_content.append({"type": "text", "text": block.text.rstrip()})
            elif hasattr(block, "type") and block.type == "tool_use":
                sanitized_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            else:
                sanitized_content.append(block)
        messages.append({"role": "assistant", "content": sanitized_content})

        if response.stop_reason == "end_turn":
            return response

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    query = getattr(block, "input", {}).get("query", "...")
                    print(f"  [{label}] Searching: {query}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "",
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"{label} phase did not complete within the iteration limit.")


def _extract_scout_json(response) -> dict:
    """
    Extracts the lightweight scout JSON from Claude's response.
    Returns a dict with top_stories, recommended_lead, and lead_rationale.
    Falls back to an empty dict if parsing fails (deep phase will still run).
    """
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", full_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{[\s\S]*\}", full_text)
        json_str = json_match.group(0) if json_match else ""

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        print("  [Scout] Could not parse scout JSON — deep phase will proceed without scout context.")
        return {}


def run_research_agent() -> dict:
    """
    Two-phase scout-then-dig research agent.

    Phase A — Scout (3 searches):
        Broad headline sweep to find the top stories of the week.
        Claude returns a lightweight JSON with ~10 stories and a recommended lead.

    Phase B — Deep Research (up to 8 searches):
        Fresh conversation. Claude is told the lead story and does a focused deep
        dive (4-5 searches) plus roundup sweeps for the other sections (3-4 searches).
        Returns the full research brief JSON.

    This approach gives the lead story much more depth than a flat 10-search sweep,
    while keeping total cost similar (~11 searches total vs ~10 before).
    """
    print("\n[Phase 1/3] Researching this week's AI & Healthcare news...")
    print("  Loading previous episode headlines for deduplication...")

    previous_headlines = load_previous_episode_headlines()
    if previous_headlines:
        print(f"  Found {len(previous_headlines)} headlines from last episode — will avoid repeats.")
    else:
        print("  No previous episode found — this appears to be the first run.")

    # ── PHASE A: SCOUT ─────────────────────────────────────────────────────────
    print("  Scout phase: 3 broad searches to find the week's top stories...")

    scout_response = _run_agentic_loop(
        prompt=build_scout_prompt(previous_headlines),
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        max_tokens=2048,
        label="Scout",
    )
    scout_results = _extract_scout_json(scout_response)

    lead = scout_results.get("recommended_lead", "top story of the week")
    print(f"  Recommended lead: {lead}")

    # ── PHASE B: DEEP RESEARCH ─────────────────────────────────────────────────
    print("  Deep phase: 4-5 searches on lead story + 3-4 roundup sweeps...")

    deep_response = _run_agentic_loop(
        prompt=build_deep_research_prompt(scout_results),
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        max_tokens=MAX_TOKENS,
        label="Deep",
    )

    print("  Research complete.")
    return _extract_research_json(deep_response)


def _extract_research_json(response) -> dict:
    """
    Extracts and parses the JSON research brief from Claude's final response.
    Claude is instructed to return raw JSON — this function finds and parses it.
    """
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    # Strip markdown code fences if Claude wrapped the JSON in them.
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", full_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find a raw JSON object.
        json_match = re.search(r"\{[\s\S]*\}", full_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(
                "Could not extract JSON from research response.\n"
                f"Full response:\n{full_text[:500]}..."
            )

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Research JSON is malformed: {e}\n\nRaw JSON:\n{json_str[:500]}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CONTENT WRITING
# Uses Claude to transform the research brief into polished written content.
# ─────────────────────────────────────────────────────────────────────────────

def _claude_create_with_retry(**kwargs) -> object:
    """
    Calls claude.messages.create() with automatic retry on transient errors.

    Handles two types of errors:
    - RateLimitError (HTTP 429): hit the token-per-minute cap — wait 60s and retry.
    - APIConnectionError: server dropped the connection (network blip) — wait 10s and retry.

    Both are retried up to 3 times total before giving up.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            return claude.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == max_retries:
                raise
            wait_seconds = 60 * attempt  # 60s, then 120s
            print(f"  [Rate limit] Waiting {wait_seconds}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_seconds)
        except anthropic.APIConnectionError:
            if attempt == max_retries:
                raise
            print(f"  [Connection error] Waiting 15s before retry {attempt + 1}/{max_retries}...")
            time.sleep(15)


def write_blog_post(research: dict) -> str:
    """
    Takes the research brief and generates a complete newsletter blog post.
    This is a pure writing task — no web search needed.

    Returns:
        The full blog post as a markdown string.
    """
    print("[Phase 2/3] Writing newsletter blog post...")

    response = _claude_create_with_retry(
        model=MODEL_WRITING,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": BLOG_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": build_blog_prompt(research)}],
    )

    blog_post = ""
    for block in response.content:
        if hasattr(block, "text"):
            blog_post += block.text

    print(f"  Blog post written ({len(blog_post.split())} words).")
    return blog_post.strip()


def write_metadata(research: dict) -> dict:
    """
    Generates a short episode title and 2-4 sentence description for RSS/podcast distribution.
    Called before the podcast script is written — uses very few tokens.

    Returns:
        A dict with 'title' (5-10 words) and 'description' (2-4 sentences) keys.
    """
    episode_angle = research.get("episode_angle", "")

    # Pull the top headlines to give the metadata writer enough context.
    compact = _compact_research(research)
    top_headlines = [s.get("headline", "") for s in compact.get("ai_pulse", [])][:4]
    headlines_text = "\n".join(f"  - {h}" for h in top_headlines)

    response = _claude_create_with_retry(
        model=MODEL_WRITING,
        max_tokens=300,
        messages=[{"role": "user", "content": f"""Based on this week's healthcare AI episode, write:
1. A punchy episode title (5-10 words, headline style — no "Ep." prefix)
2. A 2-4 sentence episode description for podcast apps (shown in Apple Podcasts, Spotify, etc.)
   The description should hook the listener and summarize what they'll learn.

Episode angle: {episode_angle}

Top stories this week (for context):
{headlines_text}

Return only valid JSON in this exact format, no preamble:
{{"title": "...", "description": "..."}}"""}],
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    # Extract the JSON object from the response
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback if parsing fails — use truncated angle as title
    return {"title": episode_angle[:80], "description": episode_angle}


def write_podcast_script(research: dict) -> str:
    """
    Takes the research brief and generates a full podcast script
    optimized for ElevenLabs TTS (natural speech patterns, spoken cadence).

    Returns:
        The full podcast script as a plain text string.
    """
    print("[Phase 2/3] Writing podcast script (first pass)...")

    response = _claude_create_with_retry(
        model=MODEL_WRITING,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": PODCAST_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": build_podcast_prompt(research)}],
    )

    script = ""
    for block in response.content:
        if hasattr(block, "text"):
            script += block.text

    # Prepend the fixed show open ONLY if a real recorded intro is not available.
    # If music/intro_voice.mp3 exists, the host's real voice handles the intro in production
    # (publish.py plays it before the [MUSIC: SHOW OPEN] marker), so we skip the text version
    # to avoid a redundant intro in Notion and the podcast script.
    intro_voice_file = Path(__file__).parent.parent / "music" / "intro_voice.mp3"
    if intro_voice_file.exists():
        # Real intro on file — script starts directly with LEAD: / [MUSIC: SHOW OPEN] / hook.
        script = script.strip()
    else:
        # No recorded intro — prepend the ElevenLabs-generated show open as before.
        show_open = (
            "Welcome to Vitals & Signals. I'm Aahlay. Every week I track what's happening "
            "at the intersection of AI and healthcare — the model releases and infrastructure "
            "moves that matter for clinicians and health systems, where the money is flowing, "
            "and what's coming down the regulatory and research pipeline. "
            "Let's get into this week's episode."
        )
        script = show_open + "\n\n" + script.strip()

    word_count = len(script.split())
    est_minutes = round(word_count / 150)  # ~150 words per minute for natural speech
    print(f"  Podcast script written ({word_count} words, ~{est_minutes} min).")
    return script


def polish_podcast_script(script: str) -> str:
    """
    Second editorial pass — reads the generated script and rewrites for naturalness.

    The first-pass script is structurally sound but often reads like polished prose
    rather than spoken word. This pass removes AI-isms, loosens formal phrasing,
    and makes the host sound like a real person thinking out loud — not reading copy.

    Returns:
        The polished script as a plain text string.
    """
    print("  Polishing script for naturalness (second pass)...")

    polish_prompt = f"""You are an experienced podcast editor. Below is a first-draft podcast script.
Your job is to do a naturalness edit — make it sound like a real person speaking, not polished AI copy.

WHAT TO LOOK FOR AND FIX:

1. AI CLICHÉS — remove or rephrase these patterns:
   - "it's worth noting", "it's important to note", "it's worth mentioning"
   - "delve into", "dive into" (prefer "get into", "dig into", "look at")
   - "in conclusion", "to summarize", "in summary"
   - "certainly", "indeed", "absolutely" as filler affirmations
   - "the fact that" (usually cuttable)
   - "at the end of the day", "when all is said and done"
   - "a testament to", "a perfect example of"
   - Triple structures that sound like bullet points ("X, Y, and Z — these are the things that...")

2. OVERUSED STRUCTURAL PATTERNS — find and break up:
   - Triple negation reveals used more than once: "Not X. Not Y. Not Z. [reveal]."
     If this structure appears more than once, rephrase the extras using contrast or direct statement.
   - Paragraph openers: if "Now" or "Here's" starts more than 2 paragraphs in any section,
     rewrite some openers to start with a date, name, number, or scene instead.
   - Telegraphed emphasis: "Read that sentence again." / "Sit with that." — cut or rephrase.
   - Three or more consecutive sentences starting with the same word ("Not...", "The...", etc.)

3. OVERLY FORMAL TRANSITIONS — loosen these:
   - "Furthermore", "Moreover", "Additionally" → "And", "Plus", "On top of that"
   - "However" at the start of every pivot → "But", "Here's the thing", "And yet"
   - "It is clear that", "It becomes evident" → just state the point directly

4. RHYTHM ISSUES — natural speech stumbles and self-corrects:
   - If you see 3+ sentences of similar length in a row, vary one
   - Add one or two moments where the host seems to be thinking mid-sentence
     (use "— well, actually" or "... look, the short version is")
   - Check that rhetorical questions don't all land at the same beat length

5. NEWSLETTER VOICE LEAKAGE — the host is talking, not writing:
   - Remove any phrase that would only make sense in text
   - "As we'll see below", "outlined above", "as mentioned" → cut or rephrase
   - Numbers should be spoken ("sixteen percent", "forty-five million dollars")

6. PRESERVE COMPLETELY:
   - All music markers: [MUSIC: SHOW OPEN], [MUSIC: STORY OPEN], etc.
   - All <break time="1.5s" /> tags
   - The LEAD: line at the very top
   - The overall structure, all facts, all names, all numbers
   - The host's strong opinions and direct voice

Return ONLY the polished script — no commentary, no preamble, no diff. Just the full revised script.

---

SCRIPT TO POLISH:
{script}"""

    response = _claude_create_with_retry(
        model=MODEL_WRITING,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": polish_prompt}],
    )

    polished = ""
    for block in response.content:
        if hasattr(block, "text"):
            polished += block.text

    polished = polished.strip()
    word_count = len(polished.split())
    est_minutes = round(word_count / 150)
    print(f"  Polished script ({word_count} words, ~{est_minutes} min).")
    return polished


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: SAVE OUTPUT
# Saves everything to local files before pushing to Notion.
# ─────────────────────────────────────────────────────────────────────────────

def save_output(
    episode_num: int,
    research: dict,
    podcast_script: str,
    metadata: dict,
    blog_post: str = "",
) -> Path:
    """
    Saves the research brief, podcast script, and metadata to local files.
    Files are saved in: output/ep{N}-{YYYY-MM-DD}/
    blog_post is optional — skipped if empty.

    Returns:
        The Path to the output directory for this episode.
    """
    ep_dir = OUTPUT_DIR / f"ep{episode_num:03d}-{TODAY.isoformat()}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    (ep_dir / "research_brief.json").write_text(
        json.dumps(research, indent=2), encoding="utf-8"
    )
    if blog_post:
        (ep_dir / "blog_post.md").write_text(blog_post, encoding="utf-8")
    (ep_dir / "podcast_script.txt").write_text(podcast_script, encoding="utf-8")
    (ep_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"  Output saved to: {ep_dir}")
    return ep_dir


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: NOTION INTEGRATION
# Pushes the content to your Notion database as a new Draft page.
# ─────────────────────────────────────────────────────────────────────────────

def get_next_episode_number() -> int:
    """
    Queries the Notion database to find the highest existing episode number,
    then returns that number + 1.
    """
    try:
        result = notion.databases.query(
            database_id=NOTION_DB_ID,
            sorts=[{"property": "Episode Number", "direction": "descending"}],
            page_size=1,
        )
        pages = result.get("results", [])
        if not pages:
            return 1

        ep_num_prop = pages[0]["properties"].get("Episode Number", {})
        existing_num = ep_num_prop.get("number") or 0
        return int(existing_num) + 1

    except Exception as e:
        print(f"  [Warning] Could not query Notion for episode number: {e}")
        print("  Defaulting to episode 1.")
        return 1


def _split_text(text: str, max_length: int = 1800) -> list:
    """
    Splits a long string into chunks no longer than max_length characters.
    Tries to split at sentence boundaries to keep text readable.
    (Notion caps text blocks at 2000 characters.)
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while len(text) > max_length:
        split_at = text.rfind(". ", 0, max_length)
        if split_at == -1:
            split_at = max_length
        else:
            split_at += 1
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        chunks.append(text)
    return chunks


def _parse_inline_markdown(text: str) -> list:
    """
    Converts inline markdown (**bold**) into Notion's rich_text format.
    Notion doesn't render raw markdown — it needs structured JSON text objects.
    """
    if not text:
        return [{"type": "text", "text": {"content": ""}}]

    rich_text = []
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
    Converts a markdown string into Notion block objects.
    Handles: ## headings, ### headings, - bullet lists, **bold**, ---, paragraphs.
    """
    blocks = []
    lines = markdown_text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            i += 1
            continue

        if re.match(r"^-{3,}$", line) or re.match(r"^\*{3,}$", line):
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            i += 1
            continue

        if line.startswith("## "):
            blocks.append({
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": _parse_inline_markdown(line[3:].strip())},
            })
            i += 1
            continue

        if line.startswith("### "):
            blocks.append({
                "object": "block", "type": "heading_3",
                "heading_3": {"rich_text": _parse_inline_markdown(line[4:].strip())},
            })
            i += 1
            continue

        if re.match(r"^[\-\*]\s+", line):
            item_text = re.sub(r"^[\-\*]\s+", "", line).strip()
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
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
            for chunk in _split_text(paragraph_text, max_length=1800):
                blocks.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": _parse_inline_markdown(chunk)},
                })
        continue

    return blocks


def push_to_notion(
    episode_num: int,
    research: dict,
    podcast_script: str,
    metadata: dict,
    blog_post: str = "",
) -> str:
    """
    Creates a new page in your Notion database with the episode content.

    Returns:
        The URL of the newly created Notion page.
    """
    print("[Phase 3/3] Pushing to Notion...")

    episode_angle = research.get("episode_angle", "AI & Healthcare Weekly")
    short_title = metadata.get("title", episode_angle)
    page_title = f"Ep. {episode_num:03d} — {short_title}"

    page_blocks = []

    # Episode angle as a callout at the top.
    page_blocks.append({
        "object": "block", "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": episode_angle}}],
            "icon": {"emoji": "📡"},
            "color": "blue_background",
        },
    })

    # Horizon signals as a quick-reference block at the top of the page.
    horizon = research.get("horizon", {})
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

    # Optionally include the blog post (skipped when running script-only mode).
    if blog_post:
        page_blocks.append({"object": "block", "type": "divider", "divider": {}})
        page_blocks.append({
            "object": "block", "type": "heading_1",
            "heading_1": {"rich_text": [{"type": "text", "text": {"content": "Newsletter Post"}}]},
        })
        page_blocks.extend(_markdown_to_notion_blocks(blog_post))

    # Divider, then the full podcast script.
    page_blocks.append({"object": "block", "type": "divider", "divider": {}})
    page_blocks.append({
        "object": "block", "type": "heading_1",
        "heading_1": {"rich_text": [{"type": "text", "text": {"content": "Podcast Script"}}]},
    })
    for paragraph in podcast_script.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in _split_text(paragraph, max_length=1800):
            page_blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": _parse_inline_markdown(chunk)},
            })

    # Create the Notion page — first 100 blocks with the page creation call.
    BATCH_SIZE = 100
    first_batch      = page_blocks[:BATCH_SIZE]
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

    # Append remaining blocks in batches of 100.
    for i in range(0, len(remaining_blocks), BATCH_SIZE):
        batch = remaining_blocks[i:i + BATCH_SIZE]
        notion.blocks.children.append(block_id=page_id, children=batch)
        print(f"  Appended blocks {i + BATCH_SIZE + 1}–{i + BATCH_SIZE + len(batch)} to Notion...")

    print(f"  Notion page created: {page_url}")
    return page_url


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main function — runs all phases in order and prints a final summary.

    Usage:
      python scripts/research.py
      python scripts/research.py --episode 5       # force episode number
      python scripts/research.py --no-notion       # skip Notion push
    """
    parser = argparse.ArgumentParser(
        description="Vitals & Signals — Research & Writing Pipeline"
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Override the auto-detected episode number (e.g. --episode 5)"
    )
    parser.add_argument(
        "--no-notion", action="store_true",
        help="Skip pushing to Notion (useful for testing)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use claude-sonnet-4-6 for ALL phases (research + writing). "
             "Roughly $0.15/run instead of $1.20. Great for testing prompts."
    )
    parser.add_argument(
        "--skip-research", action="store_true",
        help="Skip the research phase and load research_brief.json from the existing "
             "episode folder. Use this to re-run just the writing phases without "
             "burning API credits on a new web search. Requires --episode N or an "
             "existing output folder to read from."
    )
    args = parser.parse_args()

    # --fast overrides both models to Sonnet for a cheap, quick run.
    # We modify the module-level constants so every function picks them up.
    global MODEL_RESEARCH, MODEL_WRITING
    if args.fast:
        MODEL_RESEARCH = "claude-sonnet-4-6"
        MODEL_WRITING  = "claude-sonnet-4-6"
        print("  [Fast mode] Both phases using claude-sonnet-4-6 (~$0.15/run)")

    print("=" * 60)
    print("  VITALS & SIGNALS — Research & Writing Pipeline")
    print(f"  Week of: {TODAY.isoformat()}")
    print(f"  Research window: {TWO_WEEKS_AGO.isoformat()} to {TODAY.isoformat()}")
    print(f"  Models: research={MODEL_RESEARCH.split('-')[1]}, writing={MODEL_WRITING.split('-')[1]}")
    print("=" * 60)

    # ── Phase 1: Research (or load from file) ────────────────────────────────
    if args.skip_research:
        # --skip-research: load the existing research brief from disk instead of
        # running a new web search. Much cheaper — only the writing phases run.
        print("\n[Phase 1/3] Skipping research — loading from saved file...")

        # Find the right folder: --episode N if given, otherwise most recent.
        if args.episode:
            folder = OUTPUT_DIR / f"ep{args.episode:03d}-*"
            matches = sorted(OUTPUT_DIR.glob(f"ep{args.episode:03d}-*"))
        else:
            matches = sorted(OUTPUT_DIR.glob("ep*"), reverse=True)

        if not matches:
            sys.exit(
                "[ERROR] No existing episode folders found in output/.\n"
                "        Run without --skip-research to generate content first."
            )

        ep_dir = matches[0]
        brief_path = ep_dir / "research_brief.json"
        if not brief_path.exists():
            sys.exit(f"[ERROR] research_brief.json not found in {ep_dir}")

        research = json.loads(brief_path.read_text(encoding="utf-8"))

        # Extract episode number from folder name (e.g. "ep001-2026-02-24" -> 1)
        episode_num = int(ep_dir.name[2:5])
        print(f"  Loaded: {brief_path}")
        print(f"  Episode number: {episode_num}")

    else:
        # Normal flow: run the full research agent with web search.
        research = run_research_agent()

        # ── Determine episode number ─────────────────────────────────────────
        # Do this before writing so we can save the research brief immediately —
        # if writing fails, the research isn't lost.
        if args.episode:
            episode_num = args.episode
            print(f"\n  Using forced episode number: {episode_num}")
        elif not args.no_notion:
            episode_num = get_next_episode_number()
            print(f"\n  Auto-detected next episode number: {episode_num}")
        else:
            episode_num = 1
            print("\n  No Notion connection — defaulting to episode 1 for file naming.")

        # Save the research brief immediately so it's never lost if writing fails.
        ep_dir = OUTPUT_DIR / f"ep{episode_num:03d}-{TODAY.isoformat()}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        research_path = ep_dir / "research_brief.json"
        research_path.write_text(json.dumps(research, indent=2), encoding="utf-8")
        print(f"\n  Research brief saved to: {research_path}")

    ftm = research.get("follow_the_money", {})
    print(f"\n  Episode angle: {research.get('episode_angle', 'N/A')}")
    print(f"  Stories found:")
    print(f"    The AI Pulse:        {len(research.get('ai_pulse', []))} stories")
    print(f"    Public co. earnings: {len(ftm.get('public_companies', []))} reports")
    print(f"    Startup funding:     {len(ftm.get('startups', []))} rounds")
    print(f"    Acquisitions:        {len(ftm.get('acquisitions', []))} deals")
    print(f"    On the Radar:        {len(research.get('on_the_radar', []))} items")
    print(f"    Upcoming earnings:   {len(research.get('upcoming_earnings', []))} companies")

    # ── Phase 2: Writing ─────────────────────────────────────────────────────
    # We pause 90 seconds between phases because the research agent uses many
    # tokens (web search results are verbose). Anthropic's API has a 30,000
    # token/minute rate limit on Tier 1 accounts. The pause lets the rolling
    # window reset so writing calls go through cleanly.
    # Once your account reaches Tier 2+, these waits become unnecessary.
    print()
    print("  Pausing 90s between phases to stay within API rate limits...")
    time.sleep(90)

    # Generate short title + RSS description from the research brief directly.
    # (No blog post is generated — only the podcast script is produced.)
    metadata = write_metadata(research)
    print(f"  Episode title: {metadata.get('title', '(no title)')}")

    # Save metadata immediately — if script generation fails, this isn't lost.
    (ep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Metadata saved.")

    # Write the first-pass podcast script.
    podcast_script = write_podcast_script(research)

    # Polish the script: second pass to remove AI-isms and improve naturalness.
    print("  Pausing 30s before naturalness polish pass...")
    time.sleep(30)
    podcast_script = polish_podcast_script(podcast_script)

    # ── Save remaining files locally ──────────────────────────────────────────
    print()
    output_path = save_output(episode_num, research, podcast_script, metadata)

    # ── Phase 3: Notion ──────────────────────────────────────────────────────
    notion_url = None
    if not args.no_notion:
        print()
        notion_url = push_to_notion(episode_num, research, podcast_script, metadata)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE.")
    print(f"  Episode:      Ep. {episode_num:03d}")
    print(f"  Publish date: {PUBLISH_DATE.isoformat()}")
    print(f"  Files saved:  {output_path}")
    if notion_url:
        print(f"  Notion draft: {notion_url}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Open the Notion page and review the podcast script.")
    print("  2. Fill in the editorial feedback properties (Rating, Notes, etc.).")
    print("  3. Change Status from Draft -> Approved when ready.")
    print("  4. Run scripts/publish.py to generate audio and publish.")
    print("\n  NOTE: To update the ElevenLabs voice, change ELEVENLABS_VOICE_ID in .env")


if __name__ == "__main__":
    main()
