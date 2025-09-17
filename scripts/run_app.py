"""
Main runner for the Therapeutic Music AI Gradio app
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

def setup_environment():
    """Load environment variables and check setup"""
    
    # Load environment variables
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=True)
        print(f"✅ Loaded environment from: {env_file}")
    else:
        print("⚠️  No .env file found. Creating template...")
        from setup_env import create_env_template
        create_env_template()
        print("Please configure your .env file and run again.")
        return False
    
    # Check required keys
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    replicate_ok = bool(os.getenv("REPLICATE_API_TOKEN"))
    
    print(f"OpenAI API: {'✅' if openai_ok else '❌'}")
    print(f"Replicate API: {'✅' if replicate_ok else '❌'}")
    
    if not openai_ok:
        print("❌ OpenAI API key is required. Please set OPENAI_API_KEY in .env")
        return False
    
    if not replicate_ok:
        print("⚠️  Replicate API token not set. Music generation will be disabled.")
        print("   Set REPLICATE_API_TOKEN in .env to enable music generation.")
    
    return True

def main():
    """Main application entry point"""
    
    print("🎵 Starting Therapeutic Music AI")
    print("=" * 40)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed. Please check your .env configuration.")
        sys.exit(1)
    
    print("\n🚀 Launching Gradio interface...")
    
    try:
        # Import and run the Gradio app
        from gradio_app import create_gradio_interface
        
        demo = create_gradio_interface()
        
        print("\n✅ Gradio interface created successfully!")
        print("🌐 The app will be available at:")
        print("   - Local: http://localhost:7860")
        print("   - Network: http://0.0.0.0:7860")
        print("   - Public link will be generated automatically")
        
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True,
            show_error=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
