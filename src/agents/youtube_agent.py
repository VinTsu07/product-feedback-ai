from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
import youtube_transcript_api
from typing import Dict, Any, List, Optional
import logging
import time
import re
from datetime import datetime  
from .base_agent import BaseAgent

class YouTubeAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.youtube = self._setup_youtube_client()
        self.quota_used = 0  # Track API quota usage

    def format_output(self, data: Any) -> Dict[str, Any]:
        """Standardize output format"""
        return {
            "agent": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "status": "completed"
        }
        
    def _setup_youtube_client(self):
        """
        Initialize YouTube Data API v3 client with error handling
        
        Returns:
            googleapiclient.discovery.Resource: YouTube API client or None if failed
            
        Quota Cost: 0 units (initialization only)
        """
        try:
            if not self.config.YOUTUBE_API_KEY:
                self.logger.error("❌ YouTube API key not found in config")
                return None
            
            # Build YouTube API client
            youtube_client = build(
                'youtube', 
                'v3', 
                developerKey=self.config.YOUTUBE_API_KEY,
                cache_discovery=False  # Avoid cache warnings
            )
            
            # Test the connection with a simple quota-free call
            # We'll test this in the search function instead to avoid extra quota usage
            
            self.logger.info("✅ YouTube API client initialized successfully")
            return youtube_client
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YouTube API client: {str(e)}")
            return None
        
    def search_videos(self, product_name: str, search_query: List, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for YouTube videos related to the product with intelligent query strategy
        
        Args:
            product_name: The product being analyzed (e.g., "iPhone 15 Pro")
            search_query: Search terms from orchestrator (e.g., "iPhone 15 Pro review problems")
            max_results: Maximum number of videos to return (default: 20)
            
        Returns:
            List[Dict]: List of video data dictionaries
            
        Quota Cost: 100 units per search request
        """
        if not self.youtube:
            self.logger.error("❌ YouTube client not initialized")
            return []
        
        videos = []
        
        try:
            # Multi-query strategy for better coverage
            videos_per_query = max(1, max_results // len(search_query))
            
            for query in search_query:
                self.logger.info(f"🔍 Searching YouTube for: '{query}'")
                
                # YouTube Data API search request
                search_response = self.youtube.search().list(
                    q=query,
                    part='snippet',
                    type='video',
                    maxResults=min(videos_per_query, 50),  # API limit is 50
                    order='relevance',  # Can also try 'viewCount', 'date'
                    publishedAfter='2023-01-01T00:00:00Z',  # Only recent videos
                    videoDuration='medium',  # 4-20 minutes (filter out very short/long)
                    regionCode='US',  # Focus on English content
                    relevanceLanguage='en'
                ).execute()
                
                # Track quota usage (search costs 100 units)
                self.quota_used += 100
                self.logger.info(f"📊 API Quota used: {self.quota_used} units")
                
                # Process search results
                for item in search_response.get('items', []):
                    video_data = self._parse_search_result(item, query)
                    if video_data:
                        videos.append(video_data)
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
                # Stop if we have enough videos
                if len(videos) >= max_results:
                    break
            
            # Remove duplicates (same video from different queries)
            videos = self._deduplicate_videos(videos)
            
            # Sort by relevance score (basic for now)
            videos.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            self.logger.info(f"✅ Found {len(videos)} unique videos")
            return videos[:max_results]
            
        except HttpError as e:
            self.logger.error(f"❌ YouTube API error: {e}")
            return []
        except Exception as e:
            self.logger.error(f"❌ Video search failed: {str(e)}")
            return []

    def _parse_search_result(self, item: Dict, search_query: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single video search result into structured format
        
        Args:
            item: Raw search result item from YouTube API
            search_query: The query that found this video
            
        Returns:
            Dict: Structured video data or None if invalid
        """
        try:
            snippet = item['snippet']
            video_id = item['id']['videoId']
            
            # Basic relevance scoring (expand this later)
            title = snippet.get('title', '').lower()
            description = snippet.get('description', '').lower()
            
            # Simple keyword-based relevance (improve later with ML)
            relevance_keywords = ['review', 'problem', 'issue', 'test', 'honest', 'real']
            relevance_score = sum(1 for keyword in relevance_keywords if keyword in title or keyword in description)
            relevance_score = min(relevance_score / len(relevance_keywords), 1.0)  # Normalize to 0-1
            
            video_data = {
                'video_id': video_id,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'channel_id': snippet.get('channelId', ''),
                'channel_title': snippet.get('channelTitle', ''),
                'published_at': snippet.get('publishedAt', ''),
                'thumbnail_url': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
                'search_query_used': search_query,
                'relevance_score': relevance_score,
                'url': f"https://www.youtube.com/watch?v={video_id}"
            }
            
            return video_data
            
        except KeyError as e:
            self.logger.warning(f"⚠️ Missing expected field in video data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error parsing video result: {e}")
            return None

    def _deduplicate_videos(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate videos found by different search queries
        
        Args:
            videos: List of video dictionaries
            
        Returns:
            List[Dict]: Deduplicated video list
        """
        seen_video_ids = set()
        unique_videos = []
        
        for video in videos:
            video_id = video.get('video_id')
            if video_id and video_id not in seen_video_ids:
                seen_video_ids.add(video_id)
                unique_videos.append(video)
        
        return unique_videos
    
    def extract_video_metadata(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract detailed metadata for videos (views, likes, duration, etc.)
        
        Args:
            videos: List of basic video data from search
            
        Returns:
            List[Dict]: Videos enriched with detailed metadata
            
        Quota Cost: 1 unit per video (for videos.list call)
        """
        if not self.youtube or not videos:
            return videos
        
        enriched_videos = []
        
        try:
            # Batch process videos to minimize API calls
            video_ids = [video['video_id'] for video in videos if 'video_id' in video]
            
            # YouTube API allows up to 50 video IDs per request
            batch_size = 50
            
            for i in range(0, len(video_ids), batch_size):
                batch_ids = video_ids[i:i + batch_size]
                
                self.logger.info(f"📊 Fetching metadata for {len(batch_ids)} videos...")
                
                # Get detailed video statistics
                videos_response = self.youtube.videos().list(
                    part='statistics,contentDetails,status',
                    id=','.join(batch_ids)
                ).execute()
                
                # Track quota (1 unit per video, but batched)
                self.quota_used += 1
                
                # Create lookup dictionary for quick access
                metadata_lookup = {}
                for video_item in videos_response.get('items', []):
                    video_id = video_item['id']
                    metadata_lookup[video_id] = self._parse_video_metadata(video_item)
                
                # Enrich original videos with metadata
                for video in videos[i:i + batch_size]:
                    if video['video_id'] in metadata_lookup:
                        enriched_video = {**video, **metadata_lookup[video['video_id']]}
                        
                        # Calculate engagement rate
                        enriched_video['engagement_rate'] = self._calculate_engagement_rate(enriched_video)
                        
                        # Update relevance score with engagement data
                        enriched_video['relevance_score'] = self._update_relevance_score(enriched_video)
                        
                        enriched_videos.append(enriched_video)
                    else:
                        # Keep original video if metadata fetch failed
                        enriched_videos.append(video)
                
                # Rate limiting
                time.sleep(0.3)
            
            self.logger.info(f"✅ Enriched {len(enriched_videos)} videos with metadata")
            return enriched_videos
            
        except HttpError as e:
            self.logger.error(f"❌ YouTube API error fetching metadata: {e}")
            return videos  # Return original videos if metadata fetch fails
        except Exception as e:
            self.logger.error(f"❌ Metadata extraction failed: {str(e)}")
            return videos

    def _parse_video_metadata(self, video_item: Dict) -> Dict[str, Any]:
        """
        Parse video metadata from YouTube API response
        
        Args:
            video_item: Video item from YouTube videos.list API
            
        Returns:
            Dict: Parsed metadata
        """
        statistics = video_item.get('statistics', {})
        content_details = video_item.get('contentDetails', {})
        status = video_item.get('status', {})
        
        # Parse ISO 8601 duration (PT4M13S -> 253 seconds)
        duration_str = content_details.get('duration', 'PT0S')
        duration_seconds = self._parse_duration(duration_str)
        
        metadata = {
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'duration_seconds': duration_seconds,
            'duration_formatted': self._format_duration(duration_seconds),
            'upload_status': status.get('uploadStatus', 'unknown'),
            'privacy_status': status.get('privacyStatus', 'unknown'),
            'license': status.get('license', 'unknown'),
            'made_for_kids': status.get('madeForKids', False)
        }
        
        return metadata

    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration string to seconds
        
        Args:
            duration_str: ISO 8601 duration (e.g., "PT4M13S")
            
        Returns:
            int: Duration in seconds
        """
        import re
        
        # Pattern to match PT[hours]H[minutes]M[seconds]S
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds

    def _format_duration(self, seconds: int) -> str:
        """Format duration seconds into human-readable format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def _calculate_engagement_rate(self, video: Dict[str, Any]) -> float:
        """
        Calculate engagement rate (likes + comments) / views
        
        Args:
            video: Video data with statistics
            
        Returns:
            float: Engagement rate (0.0 to 1.0)
        """
        views = video.get('view_count', 0)
        likes = video.get('like_count', 0)
        comments = video.get('comment_count', 0)
        
        if views == 0:
            return 0.0
        
        engagement = (likes + comments) / views
        return min(engagement, 1.0)  # Cap at 100%

    def _update_relevance_score(self, video: Dict[str, Any]) -> float:
        """
        Update relevance score based on basic metrics
        
        Args:
            video: Video data with metadata
            
        Returns:
            float: Updated relevance score
        """
        base_score = video.get('relevance_score', 0.0)
        
        # Simple engagement boost
        engagement_rate = video.get('engagement_rate', 0.0)
        engagement_boost = min(engagement_rate * 2, 0.2)  # Max boost of 0.2
        
        final_score = base_score + engagement_boost
        return max(0.0, min(1.0, final_score))  # Keep between 0 and 1
    
    def extract_video_transcripts(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract transcripts/captions from videos for detailed analysis
        
        Args:
            videos: List of video data
            
        Returns:
            List[Dict]: Videos enriched with transcript data
            
        Note: Uses youtube-transcript-api (free, no API quota)
        """
        if not videos:
            return videos
        
        self.logger.info(f"📝 Extracting transcripts from {len(videos)} videos...")
        
        for video in videos:
            video_id = video.get('video_id')
            if not video_id:
                continue
                
            try:
                # Try to get transcript in English
                transcript_list = YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=['en', 'en-US', 'en-GB']
                )
                
                # Combine all transcript segments into full text
                full_transcript = " ".join([segment['text'] for segment in transcript_list])
                
                # Store both segmented and full transcript
                video['transcript'] = {
                    'full_text': full_transcript,
                    'segments': transcript_list[:10],  # Store first 10 segments for context
                    'total_segments': len(transcript_list),
                    'language': 'en',
                    'available': True
                }
                
                self.logger.info(f"✅ Transcript extracted for: {video.get('title', 'Unknown')[:50]}...")
                
            except youtube_transcript_api._errors.TranscriptsDisabled:
                video['transcript'] = {
                    'full_text': '',
                    'segments': [],
                    'total_segments': 0,
                    'language': None,
                    'available': False,
                    'error': 'Transcripts disabled for this video'
                }
                self.logger.warning(f"⚠️ Transcripts disabled for video: {video_id}")
                
            except youtube_transcript_api._errors.NoTranscriptFound:
                video['transcript'] = {
                    'full_text': '',
                    'segments': [],
                    'total_segments': 0,
                    'language': None,
                    'available': False,
                    'error': 'No transcript found'
                }
                self.logger.warning(f"⚠️ No transcript found for video: {video_id}")
                
            except Exception as e:
                video['transcript'] = {
                    'full_text': '',
                    'segments': [],
                    'total_segments': 0,
                    'language': None,
                    'available': False,
                    'error': str(e)
                }
                self.logger.error(f"❌ Transcript extraction failed for {video_id}: {str(e)}")
            
            time.sleep(0.2)
        
        transcripts_found = sum(1 for v in videos if v.get('transcript', {}).get('available', False))
        self.logger.info(f"📝 Extracted {transcripts_found}/{len(videos)} transcripts successfully")
        
        return videos
    

    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main YouTube processing logic - orchestrates the entire workflow
        
        Args:
            task_data: Task data from OrchestratorState containing:
                - task_id: Unique task identifier
                - product_name: Product being analyzed
                - search_plan: Search configuration from planning agent
                - user_query: Original user query for context
                
        Returns:
            Dict[str, Any]: Formatted results for OrchestratorState.youtube_results
        """
        start_time = time.time()
        
        if not self.validate_input(task_data):
            return {"error": "Invalid input data", "youtube_results": []}
        
        product_name = task_data["product_name"]
        search_plan = task_data.get("search_plan", {})
        
        self.logger.info(f"🎥 Starting YouTube analysis for: {product_name}")
        
        try:
            # Phase 1: Video Discovery
            youtube_query = search_plan.get("youtube_query", [f"{product_name} review"])
            max_videos = search_plan.get("max_videos", 15)  # Reasonable default for prototype
            
            self.logger.info(f"🔍 Searching for videos with query: '{youtube_query}'")
            videos = self.search_videos(
                product_name=product_name,
                search_query=youtube_query,
                max_results=max_videos
            )
            
            if not videos:
                self.logger.warning("⚠️ No videos found")
                return {"error": "No videos found", "youtube_results": []}
            
            self.logger.info(f"📹 Found {len(videos)} videos")
            
            # Phase 2: Metadata Enrichment
            self.logger.info("📊 Enriching videos with metadata...")
            enriched_videos = self.extract_video_metadata(videos)
            
            # Phase 3: Quality Filtering (basic for prototype)
            filtered_videos = self._filter_videos_for_quality(enriched_videos)
            
            self.logger.info(f"✅ {len(filtered_videos)} videos passed quality filters")
            
            # Phase 4: Transcript Extraction (for detailed analysis)
            self.logger.info(f"📝 Extracting transcripts for detailed analysis...")
            videos_with_transcripts = self.extract_video_transcripts(filtered_videos)
            
            # Phase 6: Results Processing
            processing_time = time.time() - start_time
            final_results = self._prepare_final_results(
                videos_with_transcripts, 
                task_data, 
                processing_time
            )
            
            self.logger.info(f"🎯 YouTube analysis complete in {processing_time:.2f}s")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ YouTube processing failed: {str(e)}")
            return self._format_error_results(task_data, str(e))

    def _filter_videos_for_quality(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply basic quality filters to remove low-quality videos
        
        Args:
            videos: List of enriched video data
            
        Returns:
            List[Dict]: Filtered video list
        """
        filtered = []
        
        for video in videos:
            # Skip videos with very low views (likely not useful)
            if video.get('view_count', 0) < 100:
                continue
                
            # Skip very short videos (likely not reviews)
            if video.get('duration_seconds', 0) < 60:
                continue
                
            # Skip extremely long videos (likely not focused reviews)
            if video.get('duration_seconds', 0) > 3600:  # 1 hour
                continue
                
            # Skip videos with very low relevance scores
            if video.get('relevance_score', 0) < 0.1:
                continue
                
            # Skip videos that are made for kids (likely not product reviews)
            if video.get('made_for_kids', False):
                continue
                
            filtered.append(video)
        
        return filtered

    def _prepare_final_results(self, videos: List[Dict[str, Any]], task_data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """
        Format final results for the orchestrator state
        
        Args:
            videos: Processed video data
            task_data: Original task data
            processing_time: Time taken for processing
            
        Returns:
            Dict: Formatted results matching OrchestratorState expectations
        """
        # Calculate summary statistics
        total_videos = len(videos)
        total_views = sum(video.get('view_count', 0) for video in videos)
        total_comments = sum(video.get('comment_count', 0) for video in videos)
        avg_relevance = sum(video.get('relevance_score', 0) for video in videos) / max(total_videos, 1)
        
        return self.format_output({
            "youtube_results": videos,
            "summary_stats": {
                "total_videos_analyzed": total_videos,
                "total_views_across_videos": total_views,
                "total_comments_available": total_comments,
                "average_relevance_score": round(avg_relevance, 3),
                "api_quota_used": self.quota_used
            },
            "processing_metadata": {
                "processing_time_seconds": round(processing_time, 2),
                "product_analyzed": task_data["product_name"],
                "search_strategy_used": task_data.get("search_plan", {}).get("youtube_query", ""),
                "timestamp": datetime.now().isoformat()
            },
            "status": "completed"
        })

    def _format_empty_results(self, task_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Format empty results when no videos are found"""
        return self.format_output({
            "youtube_results": [],
            "summary_stats": {
                "total_videos_analyzed": 0,
                "reason": reason,
                "api_quota_used": self.quota_used
            },
            "status": "completed_empty"
        })

    def _format_error_results(self, task_data: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Format error results when processing fails"""
        return {
            "error": error_message,
            "youtube_results": [],
            "status": "failed",
            "api_quota_used": self.quota_used
        }