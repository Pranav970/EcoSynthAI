import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class DataLogger:
    def __init__(
        self,
        log_dir: str = 'logs/',
        project_name: str = 'ecosynthai'
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file handler
        log_file = self.log_dir / f"{project_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(project_name)

    def log_experiment(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts_path: Optional[str] = None
    ):
        try:
            experiment_data = {
                'timestamp': datetime.now().isoformat(),
                'name': experiment_name,
                'parameters': parameters,
                'metrics': metrics,
                'artifacts_path': artifacts_path
            }
            
            # Save to JSON
            experiment_file = self.log_dir / f"experiment_{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}