from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the agent"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results"""
        pass
    
    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["task_id", "product_name"]
        return all(field in task_data for field in required_fields)
