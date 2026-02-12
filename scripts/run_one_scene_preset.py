#!/usr/bin/env python3
"""
Run exactly ONE scene per execution (user-selected preset).

This is the recommended way when you want:
- each run only processes one scene
- user explicitly chooses which scene to run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PRESETS = {
    "场景1_工作与生产力": {
        "scene": "工作与生产力",
        "keywords": [
            "productivity",
            "work productivity",
            "focus at work",
            "deep work",
            "procrastination",
            "time management",
            "task management",
            "getting things done",
            "gtd",
            "pomodoro",
            "context switching",
            "meeting fatigue",
            "information overload",
            "work burnout",
            "ADHD productivity",
        ],
    },
    "场景2_运动与体能训练": {
        "scene": "运动与体能训练",
        "keywords": [
            "workout recovery",
            "post workout recovery",
            "muscle soreness",
            "DOMS recovery",
            "strength training",
            "hypertrophy",
            "cardio endurance",
            "zone 2 training",
            "HIIT",
            "overtraining",
            "training fatigue",
            "lifting plateau",
            "rest day",
            "deload week",
            "injury prevention",
        ],
    },
    "场景3_睡眠与恢复": {
        "scene": "睡眠与恢复",
        "keywords": [
            "sleep quality",
            "improve sleep",
            "insomnia",
            "can't fall asleep",
            "can't stay asleep",
            "waking up tired",
            "sleep hygiene",
            "sleep schedule",
            "circadian rhythm",
            "blue light sleep",
            "melatonin",
            "magnesium for sleep",
            "sleep apnea",
            "sleep debt",
            "stress insomnia",
        ],
    },
    "场景4_心理健康与情绪": {
        "scene": "心理健康与情绪",
        "keywords": [
            "anxiety",
            "depression",
            "stress",
            "burnout",
            "panic attacks",
            "rumination",
            "overthinking",
            "emotional regulation",
            "therapy",
            "CBT",
            "mindfulness",
            "work stress",
            "high functioning anxiety",
            "imposter syndrome",
            "SSRI side effects",
        ],
    },
    "场景5_特殊职业_夜班工作": {
        "scene": "特殊职业（如夜班工作）",
        "keywords": [
            "night shift",
            "night shift fatigue",
            "shift work",
            "rotating shifts",
            "shift work sleep disorder",
            "sleep after night shift",
            "staying awake night shift",
            "night shift meal timing",
            "caffeine night shift",
            "napping night shift",
            "circadian rhythm night shift",
            "daytime sleep",
            "blackout curtains",
            "shift worker burnout",
            "night shift weight gain",
        ],
    },
    "场景6_慢性病或健康问题管理": {
        "scene": "慢性病或健康问题管理",
        "keywords": [
            "chronic fatigue",
            "chronic fatigue syndrome",
            "long covid",
            "fibromyalgia",
            "chronic pain",
            "pain management",
            "autoimmune disease fatigue",
            "hypothyroid fatigue",
            "diabetes management",
            "blood sugar crash",
            "IBS management",
            "migraine triggers",
            "POTS fatigue",
            "brain fog chronic illness",
            "managing symptoms while working",
        ],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Run ONE scene preset (direct Reddit mode).")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()), help="Which scene preset to run.")
    p.add_argument("--min-request-interval", type=float, default=2.0, help="Env MIN_REQUEST_INTERVAL (seconds).")
    p.add_argument("--target-posts", type=int, default=0, help="Stop after collecting at least N posts (0=no limit).")
    p.add_argument("--max-pages-per-keyword", type=int, default=120, help="Max search pages per keyword.")
    p.add_argument("--expand-more", action="store_true", default=True, help="Expand morechildren for comment tree.")
    p.add_argument("--no-expand-more", action="store_false", dest="expand_more", help="Disable morechildren expansion.")
    p.add_argument("--fresh", action="store_true", help="Delete output directory first.")
    return p.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]

    crawler_script = PROJECT_ROOT / "scripts" / "crawl_reddit_scene.py"
    if not crawler_script.exists():
        raise SystemExit(f"Missing crawler script: {crawler_script}")

    env = os.environ.copy()
    env["MIN_REQUEST_INTERVAL"] = str(args.min_request_interval)
    # Direct Reddit mode: ensure PullPush is off unless user explicitly overrides.
    env.pop("USE_PULLPUSH", None)

    keywords = ",".join(preset["keywords"])

    cmd = [
        sys.executable,
        "-u",
        str(crawler_script),
        "--scene",
        preset["scene"],
        "--scene-name",
        args.preset,
        "--keywords",
        keywords,
        "--target-posts",
        str(args.target_posts),
        "--max-pages-per-keyword",
        str(args.max_pages_per_keyword),
        "--comment-sorts",
        "best,top",
    ]
    if args.fresh:
        cmd.append("--fresh")
    if args.expand_more:
        cmd.append("--expand-more")
    else:
        cmd.append("--no-expand-more")

    raise SystemExit(subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()

