# generate_openapi.py
import sys
import os
import yaml

# Load .env file so environment variables (e.g. DATABASE_URL) are available
from dotenv import load_dotenv
load_dotenv()

# Add the back directory to the system path so internal app imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "back"))

from app.main import app

# Get the OpenAPI schema from the FastAPI app
openapi_schema = app.openapi()

# Save the OpenAPI schema to a YAML file
with open('docs/churnPrediction_openapi.yaml', 'w') as f:
    yaml.dump(openapi_schema, f, indent=2)

print("OpenAPI schema has been generated and saved to docs/churnPrediction_openapi.yaml")