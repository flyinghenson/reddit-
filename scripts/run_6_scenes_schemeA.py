#!/usr/bin/env python3
"""
Run 6 scenes sequentially (scheme A).

This script is designed to be used while another long-running crawl is active,
so it defaults to running ONE scene at a time to keep total crawler concurrency low.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import urllib.request
import urllib.parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SCENES = [
    {
        "scene": "工作与生产力",
        "scene_name": "场景1_工作与生产力",
        "keywords": [
            "productivity",
            "work productivity",
            "focus at work",
            "staying focused",
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
            "work from home productivity",
            "remote work productivity",
            "burnout",
            "work burnout",
            "mental fatigue",
            "brain fog",
            "ADHD productivity",
            "executive dysfunction",
            "weekly planning",
            "daily routine",
            "habit building",
            "motivation problems",
            "discipline vs motivation",
            "staying organized",
            "overwhelmed at work",
            "work stress productivity",
        ],
    },
    {
        "scene": "运动与体能训练",
        "scene_name": "场景2_运动与体能训练",
        "keywords": [
            "workout recovery",
            "post workout recovery",
            "muscle soreness",
            "DOMS recovery",
            "strength training",
            "hypertrophy",
            "cardio endurance",
            "improve endurance",
            "running stamina",
            "zone 2 training",
            "HIIT",
            "overtraining",
            "training fatigue",
            "lifting plateau",
            "progressive overload",
            "rest day",
            "deload week",
            "injury prevention",
            "knee pain running",
            "shoulder pain lifting",
            "mobility routine",
            "stretching vs mobility",
            "warm up routine",
            "protein intake",
            "creatine",
            "electrolytes workout",
            "pre workout",
            "pre workout without caffeine",
            "sleep and recovery",
            "heart rate training",
        ],
    },
    {
        "scene": "睡眠与恢复",
        "scene_name": "场景3_睡眠与恢复",
        "keywords": [
            "sleep quality",
            "improve sleep",
            "insomnia",
            "sleep maintenance insomnia",
            "can't stay asleep",
            "can't fall asleep",
            "waking up tired",
            "morning grogginess",
            "sleep hygiene",
            "sleep schedule",
            "circadian rhythm",
            "blue light sleep",
            "melatonin",
            "melatonin timing",
            "magnesium for sleep",
            "sleep apnea",
            "snoring sleep apnea",
            "restless sleep",
            "vivid dreams tired",
            "night sweats sleep",
            "naps ruin sleep",
            "sleep debt",
            "how to wake up early",
            "shift sleep schedule",
            "stress insomnia",
            "anxiety at night",
            "sleep tracking",
            "supplements for sleep",
            "sleep and burnout",
            "recovery sleep",
        ],
    },
    {
        "scene": "心理健康与情绪",
        "scene_name": "场景4_心理健康与情绪",
        "keywords": [
            "anxiety",
            "depression",
            "stress",
            "burnout",
            "panic attacks",
            "rumination",
            "overthinking",
            "emotional regulation",
            "mood swings",
            "irritability",
            "therapy",
            "CBT",
            "mindfulness",
            "meditation for anxiety",
            "work stress",
            "high functioning anxiety",
            "imposter syndrome",
            "social anxiety",
            "loneliness",
            "motivation loss",
            "anhedonia",
            "SSRI",
            "SSRI side effects",
            "antidepressants",
            "ADHD anxiety",
            "burnout recovery",
            "how to stop worrying",
            "coping strategies",
            "self esteem",
            "sleep and anxiety",
        ],
    },
    {
        "scene": "特殊职业（如夜班工作）",
        "scene_name": "场景5_特殊职业_夜班工作",
        "keywords": [
            "night shift",
            "night shift fatigue",
            "shift work",
            "rotating shifts",
            "shift work sleep disorder",
            "sleep after night shift",
            "staying awake night shift",
            "staying alert night shift",
            "night shift meal timing",
            "night shift caffeine",
            "caffeine night shift",
            "napping night shift",
            "circadian rhythm night shift",
            "daytime sleep",
            "blackout curtains day sleep",
            "blue light night shift",
            "shift worker health",
            "shift worker burnout",
            "night shift depression",
            "night shift anxiety",
            "nurse night shift",
            "doctor night shift",
            "factory night shift",
            "24 hour shift fatigue",
            "sleep schedule night shift",
            "melatonin night shift",
            "jet lag from shift work",
            "night shift exercise",
            "insomnia after shift work",
            "night shift weight gain",
        ],
    },
    {
        "scene": "慢性病或健康问题管理",
        "scene_name": "场景6_慢性病或健康问题管理",
        "keywords": [
            "chronic fatigue",
            "chronic fatigue syndrome",
            "long covid",
            "long covid fatigue",
            "fibromyalgia",
            "chronic pain",
            "pain management",
            "autoimmune disease fatigue",
            "inflammation",
            "hypothyroid fatigue",
            "thyroid fatigue",
            "diabetes management",
            "blood sugar crash",
            "blood sugar spikes",
            "insulin resistance fatigue",
            "IBS",
            "IBS management",
            "migraine",
            "migraine triggers",
            "hypertension lifestyle",
            "high blood pressure lifestyle",
            "POTS fatigue",
            "brain fog chronic illness",
            "managing chronic illness",
            "managing symptoms while working",
            "fatigue and sleep",
            "pain and sleep",
            "flare up management",
            "medical gaslighting chronic illness",
            "chronic illness support",
        ],
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Run 6 scenes sequentially (scheme A).")
    p.add_argument(
        "--min-request-interval",
        type=float,
        default=2.0,
        help="Per-process request dispatch interval seconds (env MIN_REQUEST_INTERVAL).",
    )
    p.add_argument(
        "--target-posts",
        type=int,
        default=0,
        help="Stop after collecting at least this many posts per scene (0=no limit).",
    )
    p.add_argument(
        "--max-pages-per-keyword",
        type=int,
        default=120,
        help="Max search pages per keyword.",
    )
    p.add_argument(
        "--expand-more",
        action="store_true",
        default=True,
        help="Expand 'more' nodes via morechildren.",
    )
    p.add_argument(
        "--max-comment-posts",
        type=int,
        default=1000,
        help="Max posts to process comments for per scene (0=all). Default: 1000.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete each scene output directory first (start from scratch).",
    )
    p.add_argument(
        "--wait-pullpush",
        action="store_true",
        help="If PullPush API is unavailable, wait and retry until it is reachable.",
    )
    return p.parse_args()

def _pullpush_preflight(base_url: str) -> bool:
    url = base_url.rstrip("/") + "/reddit/search/submission/?" + urllib.parse.urlencode({"q": "test", "size": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            code = getattr(r, "status", 200)
            return 200 <= int(code) < 300
    except Exception:
        return False


def main():
    args = parse_args()

    crawler_script = PROJECT_ROOT / "scripts" / "crawl_reddit_scene.py"
    if not crawler_script.exists():
        raise SystemExit(f"Missing crawler script: {crawler_script}")

    env = os.environ.copy()
    env["MIN_REQUEST_INTERVAL"] = str(args.min_request_interval)
    env["USE_PULLPUSH"] = "1"
    pullpush_base = env.get("PULLPUSH_BASE_URL", "https://api.pullpush.io")

    # Preflight: avoid running all scenes into immediate 502s.
    if not _pullpush_preflight(pullpush_base):
        msg = f"[ERROR] PullPush API unavailable: {pullpush_base} (search endpoints failing)."
        if not args.wait_pullpush:
            print(msg, flush=True)
            print("        Re-run later, or pass --wait-pullpush to keep retrying.", flush=True)
            raise SystemExit(2)
        else:
            print(msg + " Waiting for recovery...", flush=True)
            import time
            while not _pullpush_preflight(pullpush_base):
                time.sleep(30)
            print("[OK] PullPush API is reachable. Start crawling.", flush=True)

    for idx, s in enumerate(SCENES, start=1):
        scene = s["scene"]
        scene_name = s["scene_name"]
        keywords = ",".join(s["keywords"])

        print(f"\n=== [{idx}/{len(SCENES)}] START: {scene_name} ===", flush=True)
        cmd = [
            sys.executable,
            "-u",
            str(crawler_script),
            "--scene",
            scene,
            "--scene-name",
            scene_name,
            "--keywords",
            keywords,
            "--target-posts",
            str(args.target_posts),
            "--max-pages-per-keyword",
            str(args.max_pages_per_keyword),
            "--comment-sorts",
            "best,top",
            "--max-comment-posts",
            str(args.max_comment_posts),
        ]
        if args.fresh:
            cmd.append("--fresh")
        if args.expand_more:
            cmd.append("--expand-more")
        else:
            cmd.append("--no-expand-more")

        # Run one scene to completion before moving to the next.
        # This keeps total crawler concurrency low if another crawl is already running.
        p = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
        if p.returncode != 0:
            print(f"[WARN] Scene failed: {scene_name} (exit={p.returncode}). Continue.", flush=True)

        print(f"=== [{idx}/{len(SCENES)}] DONE: {scene_name} ===", flush=True)


if __name__ == "__main__":
    main()

