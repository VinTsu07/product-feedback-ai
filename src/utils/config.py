import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dev_user:dev_pass@localhost:5432/feedback_db")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Keys
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ProductFeedbackBot/1.0")
    
    # Processing limits
    MAX_REDDIT_POSTS = int(os.getenv("MAX_REDDIT_POSTS", "50"))
    MAX_YOUTUBE_VIDEOS = int(os.getenv("MAX_YOUTUBE_VIDEOS", "20"))
    MAX_COMMENTS_PER_VIDEO = int(os.getenv("MAX_COMMENTS_PER_VIDEO", "50"))
    PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "1800"))

config = Config()
