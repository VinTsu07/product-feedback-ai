#!/usr/bin/env python3
"""
Detailed YouTube Agent Results Viewer
"""
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.youtube_agent import YouTubeAgent
from src.utils.config import config

def test_detailed_results():
    print("🧪 Detailed YouTube Agent Test")
    print("=" * 50)
    
    agent = YouTubeAgent(config)
    
    test_task = {
        "task_id": "detailed-test",
        "product_name": "iPhone 15",
        "search_plan": {
            "youtube_query": "iPhone 15 problems issues",
            "max_videos": 5
        }
    }
    
    print(f"🔍 Analyzing: {test_task['product_name']}")
    result = agent.process_task(test_task)
    
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return
    
    # Extract detailed data
    data = result.get("data", {})
    videos = data.get("youtube_results", [])
    stats = data.get("summary_stats", {})
    insights = data.get("preliminary_insights", {})
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Videos analyzed: {stats.get('total_videos_analyzed', 0)}")
    print(f"   Total views: {stats.get('total_views_across_videos', 0):,}")
    print(f"   Average relevance: {stats.get('average_relevance_score', 0)}")
    print(f"   API quota used: {stats.get('api_quota_used', 0)} units")
    
    print(f"\n🎥 DETAILED VIDEO RESULTS:")
    for i, video in enumerate(videos):
        print(f"\n   Video {i+1}:")
        print(f"      Title: {video.get('title', 'No title')}")
        print(f"      Channel: {video.get('channel_title', 'Unknown')}")
        print(f"      Views: {video.get('view_count', 0):,}")
        print(f"      Duration: {video.get('duration_formatted', 'Unknown')}")
        print(f"      Relevance Score: {video.get('relevance_score', 0):.3f}")
        print(f"      Engagement Rate: {video.get('engagement_rate', 0):.4f}")
        print(f"      URL: {video.get('url', '')}")
        print(f"      Description Preview: {video.get('description', '')[:100]}...")

        transcript = video.get('transcript', {})
        if transcript.get('available'):
            preview = transcript.get('full_text', '')[:200]
            print(f"      Transcript Preview: {preview}...")
            print(f"      Transcript Segments: {transcript.get('total_segments', 0)}")
        else:
            print(f"      Transcript: Not available ({transcript.get('error', 'Unknown error')})")
    
    print(f"\n⚠️ POTENTIAL PAIN POINTS FOUND:")
    pain_points = insights.get("potential_pain_points", [])
    for point in pain_points:
        print(f"   • '{point.get('keyword', '')}' in: {point.get('video_title', '')}")
    
    print(f"\n✨ POSITIVE SIGNALS:")
    positive = insights.get("positive_signals", [])
    for signal in positive:
        print(f"   • '{signal.get('keyword', '')}' in: {signal.get('video_title', '')}")
    
    print(f"\n👥 REVIEWER TYPES:")
    reviewer_types = insights.get("reviewer_types", {})
    print(f"   Tech Reviewers: {reviewer_types.get('tech_reviewers', 0)}")
    print(f"   General Users: {reviewer_types.get('general_users', 0)}")
    
    # Save detailed results to file
    with open('youtube_results_detailed.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full results saved to: youtube_results_detailed.json")


if __name__ == "__main__":
    test_detailed_results()

"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your agent
from src.agents.youtube_agent import YouTubeAgent
from src.utils.config import config

def test_youtube():
    print("🧪 Testing YouTube Agent")
    print("=" * 40)
    
    # Create agent
    agent = YouTubeAgent(config)
    
    # Check if API key is working
    if not agent.youtube:
        print("❌ FAILED: No YouTube API client")
        print("   Check your YOUTUBE_API_KEY in .env")
        return
    
    # Test data
    test_task = {
        "task_id": "test-123",
        "product_name": "iPhone 15",
        "search_plan": {
            "youtube_query": "iPhone 15 review",
            "max_videos": 5
        }
    }
    
    print(f"🔍 Searching for: {test_task['product_name']}")
    
    # Run the agent
    result = agent.process_task(test_task)
    
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return
    
    # Show results
    data = result.get("data", {})
    videos = data.get("youtube_results", [])
    stats = data.get("summary_stats", {})
    
    print(f"\n✅ SUCCESS!")
    print(f"📹 Videos found: {len(videos)}")
    print(f"📊 API quota used: {stats.get('api_quota_used', 0)}")
    
    # Show first video
    if videos:
        v = videos[0]
        print(f"\n🎥 Top video:")
        print(f"   Title: {v.get('title', 'No title')}")
        print(f"   Views: {v.get('view_count', 0):,}")
        print(f"   URL: {v.get('url', '')}")

if __name__ == "__main__":
    test_youtube()
"""