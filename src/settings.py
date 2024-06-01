import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(verbose=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
