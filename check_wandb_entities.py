import wandb
import os

# Load env file manually since we're running a standalone script
if os.path.exists(".wandb.env"):
    with open(".wandb.env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

print(f"Checking WandB entities for API key: {os.environ.get('WANDB_API_KEY', 'Not found')[:5]}...")

try:
    api = wandb.Api()
    print(f"Logged in user: {api.viewer.username}")
    print("Available entities (Teams):")
    for team in api.viewer.teams:
        print(f" - {team}")
    
    print("\nAlso checking api.teams():")
    try:
        # Some versions use this
        teams = api.teams()
        for t in teams:
            print(f" - {t}")
    except:
        pass
    
    print("Default entity:", api.default_entity)
    
except Exception as e:
    print(f"Error: {e}")
