# Product Feedback AI

Multi-agent system for analyzing product feedback from Reddit and YouTube.

## Quick Start

1. Clone the repository
2. Set up Python environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy environment file: `cp .env.example .env` (add your API keys)
5. Start infrastructure: `docker-compose up -d`
6. Run API: `python -m src.api.main`

## Architecture

- **Orchestrator Agent**: Manages workflow and task distribution
- **Reddit Agent**: Scrapes Reddit discussions and comments
- **YouTube Agent**: Analyzes YouTube videos and comments  
- **Analysis Agent**: Processes data and extracts insights

## Work Division

- **Person A**: Orchestration + Reddit Agent
- **Person B**: YouTube + Analysis Agent
