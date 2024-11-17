
def load_env():
    from dotenv import load_dotenv, find_dotenv
    # Initialize models and RAG
    env_file = find_dotenv()
    print(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)