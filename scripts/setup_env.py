"""
Environment setup helper for the therapeutic music AI
"""
import os
from pathlib import Path

def create_env_template():
    """Create a .env template file with required environment variables"""
    
    env_template = """# Therapeutic Music AI - Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key (required for emotion analysis and music brief generation)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Replicate API Token (required for music generation)
# Get from: https://replicate.com/account/api-tokens
REPLICATE_API_TOKEN=your_replicate_api_token_here

# Set to "1" to enable actual music generation (uses Replicate credits)
# Set to "0" to only do analysis without generating music
USE_REPLICATE=0
"""
    
    env_file = Path(".env.template")
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_template)
    
    print(f"✅ Created {env_file}")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env")
    print("2. Fill in your API keys in the .env file")
    print("3. Set USE_REPLICATE=1 when you want to generate actual music")
    print("\n🔑 API Key Sources:")
    print("- OpenAI: https://platform.openai.com/api-keys")
    print("- Replicate: https://replicate.com/account/api-tokens")

def check_environment():
    """Check current environment setup"""
    
    print("🔍 Environment Check:")
    print("-" * 40)
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        print("   Run this script to create .env.template")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    use_replicate = os.getenv("USE_REPLICATE", "0")
    
    print(f"OpenAI API Key: {'✅ Set' if openai_key else '❌ Not set'}")
    print(f"Replicate Token: {'✅ Set' if replicate_token else '❌ Not set'}")
    print(f"Music Generation: {'🎵 Enabled' if use_replicate == '1' else '🔍 Analysis only'}")
    
    if not openai_key:
        print("\n⚠️  OpenAI API key is required for emotion analysis")
    if not replicate_token:
        print("⚠️  Replicate token is required for music generation")
    
    return bool(openai_key), bool(replicate_token)

if __name__ == "__main__":
    print("🎵 Therapeutic Music AI - Environment Setup")
    print("=" * 50)
    
    # Create template if it doesn't exist
    if not Path(".env.template").exists():
        create_env_template()
    
    print()
    check_environment()
