import os

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ModuleNotFoundError:
    pass


# OpenAI / LLM configuration (optional; script can run with --keywords)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


# Reddit crawling configuration (non-official JSON endpoints)
REDDIT_BASE_URL = os.getenv("REDDIT_BASE_URL", "https://old.reddit.com")
REDDIT_WWW_URL = "https://www.reddit.com"
REDDIT_SEARCH_LIMIT = int(os.getenv("REDDIT_SEARCH_LIMIT", "100"))

# Alternative non-official dataset API (use when reddit blocks scraping)
PULLPUSH_BASE_URL = os.getenv("PULLPUSH_BASE_URL", "https://api.pullpush.io")
USE_PULLPUSH = os.getenv("USE_PULLPUSH", "").strip() in ("1", "true", "True", "yes", "YES")

# Filters (defaults are permissive; tune for quality)
POST_MIN_VOTES = int(os.getenv("POST_MIN_VOTES", "0"))
POST_MIN_COMMENTS = int(os.getenv("POST_MIN_COMMENTS", "0"))
COMMENT_MIN_VOTES = int(os.getenv("COMMENT_MIN_VOTES", "0"))

# Rate limiting (dispatch interval seconds)
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "1.0"))

