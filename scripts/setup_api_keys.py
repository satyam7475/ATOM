#!/usr/bin/env python3
"""
ATOM Secure API Key Setup Script

This script safely stores your API keys in encrypted form.
- Never stores keys in plain text
- Never commits to git
- Access logged with audit trail

Usage:
    python scripts/setup_api_keys.py

Environment Variables:
    ATOM_MASTER_PASSWORD (required): Your master password for encryption
    
Optional (for automated setup):
    GEMINI_FAST_API_KEY: Gemini 1.5 Flash API key
    GEMINI_PRO_API_KEY: Gemini 2.0 Pro API key
"""

import os
import sys
import json
import getpass
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.secure_credentials import CredentialManager


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  🔐 {title}")
    print("="*70)


def print_success(msg: str):
    """Print success message."""
    print(f"✅ {msg}")


def print_error(msg: str):
    """Print error message."""
    print(f"❌ {msg}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"⚠️  {msg}")


def print_info(msg: str):
    """Print info message."""
    print(f"ℹ️  {msg}")


def get_or_prompt_password(label: str, env_var: Optional[str] = None) -> str:
    """
    Get password from environment or prompt user.
    
    Args:
        label: Display label for password
        env_var: Environment variable name (if any)
        
    Returns:
        Password string
    """
    if env_var and env_var in os.environ:
        value = os.environ[env_var].strip()
        if value:
            print_info(f"{label}: [loaded from {env_var}]")
            return value
    
    # If it's a secret, use getpass
    if "password" in label.lower() or "secret" in label.lower():
        while True:
            value = getpass.getpass(f"\n{label}: ").strip()
            if not value:
                print_error("Cannot be empty")
                continue
            
            confirm = getpass.getpass(f"Confirm {label.lower()}: ").strip()
            if value != confirm:
                print_error("Passwords do not match")
                continue
            
            return value
    else:
        # Regular input
        while True:
            value = input(f"\n{label}: ").strip()
            if not value:
                print_error("Cannot be empty")
                continue
            return value


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    api_key = api_key.strip()
    
    # Gemini keys typically start with "AIza" and are long
    if api_key.startswith("AIza") and len(api_key) > 30:
        return True, f"✅ Valid Gemini key format (length: {len(api_key)})"
    
    # Groq keys start with "gsk_"
    if api_key.startswith("gsk_") and len(api_key) > 30:
        return True, f"✅ Valid Groq key format (length: {len(api_key)})"

    # NVIDIA NIM keys start with "nvapi-"
    if api_key.startswith("nvapi-") and len(api_key) > 30:
        return True, f"✅ Valid NVIDIA NIM key format (length: {len(api_key)})"

    # Cerebras keys start with "csk-"
    if api_key.startswith("csk-") and len(api_key) > 20:
        return True, f"✅ Valid Cerebras key format (length: {len(api_key)})"

    # Claude keys (Anthropic) and OpenAI both use "sk-"; we report
    # Anthropic first since both prefixes overlap.
    if api_key.startswith("sk-ant-") and len(api_key) > 30:
        return True, f"✅ Valid Anthropic key format (length: {len(api_key)})"
    if api_key.startswith("sk-") and len(api_key) > 30:
        return True, f"✅ Valid OpenAI key format (length: {len(api_key)})"

    return False, f"⚠️  Warning: Unexpected API key format"


def setup_master_password() -> str:
    """
    Set up master password.
    
    Returns:
        Master password
    """
    print_header("Step 1: Master Password Setup")
    
    print("\n📝 Your master password protects all stored API keys.")
    print("Store it securely (e.g., in macOS Keychain or password manager).")
    print("\nMinimum recommendations:")
    print("  - At least 32 characters")
    print("  - Mix of uppercase, lowercase, numbers")
    print("  - Save a backup in secure location")
    
    # Check if already set
    existing = os.environ.get("ATOM_MASTER_PASSWORD")
    if existing:
        use_existing = input("\n🔑 Existing ATOM_MASTER_PASSWORD found. Use it? (y/n): ").strip().lower()
        if use_existing == "y":
            return existing
        else:
            use_new = input("Generate a new one? (y/n): ").strip().lower()
            if use_new == "n":
                print_error("Master password is required")
                sys.exit(1)
    
    # Generate or prompt for password
    while True:
        choice = input("\nGenerate random password? (y/n): ").strip().lower()
        if choice == "y":
            import secrets
            master_password = secrets.token_urlsafe(32)
            print(f"\n🔑 Generated Master Password:\n{master_password}")
            
            save_to_env = input("\nSave to ATOM_MASTER_PASSWORD? (y/n): ").strip().lower()
            if save_to_env == "y":
                os.environ["ATOM_MASTER_PASSWORD"] = master_password
                print_success("Master password set in environment")
            
            return master_password
        elif choice == "n":
            master_password = get_or_prompt_password(
                "Enter Master Password",
                "ATOM_MASTER_PASSWORD"
            )
            return master_password
        else:
            print_error("Invalid choice (y/n)")


def setup_api_keys(master_password: str) -> bool:
    """
    Set up API keys.
    
    Args:
        master_password: Master password for encryption
        
    Returns:
        True if successful
    """
    print_header("Step 2: API Keys Setup")
    
    try:
        cm = CredentialManager(master_password)
    except Exception as e:
        print_error(f"Failed to initialize credential manager: {e}")
        return False
    
    # Check if keys already stored
    existing = cm.list_credentials()
    if existing:
        print(f"\n📋 Existing credentials: {', '.join(existing)}")
        modify = input("Modify/add credentials? (y/n): ").strip().lower()
        if modify != "y":
            return True
    
    credentials_to_setup = [
        ("Gemini 1.5 Flash (Fast)", "gemini_fast", "GEMINI_FAST_API_KEY"),
        ("Gemini 2.0 Pro (Powerful)", "gemini_pro", "GEMINI_PRO_API_KEY"),
        ("Groq (Llama 3.x via LPU)", "groq", "GROQ_API_KEY"),
        ("NVIDIA NIM (build.nvidia.com)", "nvidia", "NVIDIA_API_KEY"),
        ("Cerebras Cloud (Llama 3.3 70B)", "cerebras", "CEREBRAS_API_KEY"),
    ]
    
    for display_name, cred_id, env_var in credentials_to_setup:
        print(f"\n{'─'*70}")
        setup = input(f"Set up {display_name}? (y/n): ").strip().lower()
        if setup != "y":
            print_info(f"Skipped {display_name}")
            continue
        
        api_key = get_or_prompt_password(
            f"{display_name} API Key",
            env_var
        )
        
        is_valid, validation_msg = validate_api_key(api_key)
        print(validation_msg)
        
        try:
            cm.set_credential(cred_id, api_key)
            print_success(f"Stored: {display_name}")
        except Exception as e:
            print_error(f"Failed to store credential: {e}")
            return False
    
    return True


def verify_setup(master_password: str) -> bool:
    """
    Verify that credentials are properly stored.
    
    Args:
        master_password: Master password
        
    Returns:
        True if verified
    """
    print_header("Step 3: Verification")
    
    try:
        cm = CredentialManager(master_password)
        credentials = cm.list_credentials()
        
        if not credentials:
            print_warning("No credentials stored yet")
            return False
        
        print(f"\n📋 Stored credentials:")
        for cred_id in credentials:
            value = cm.get(cred_id)
            if value:
                # Show masked version
                masked = value[:8] + "..." + value[-4:]
                print(f"  ✅ {cred_id}: {masked} ({len(value)} chars)")
            else:
                print(f"  ❌ {cred_id}: (failed to retrieve)")
        
        # Test credential retrieval
        print(f"\n🧪 Testing credential retrieval...")
        test_cred = "gemini_fast"
        if test_cred in credentials:
            retrieved = cm.get(test_cred)
            if retrieved:
                print_success(f"Retrieved {test_cred} successfully")
            else:
                print_error(f"Failed to retrieve {test_cred}")
                return False
        
        return True
    
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False


def show_next_steps(master_password: str):
    """Show next steps for user."""
    print_header("Next Steps")
    
    print("\n1️⃣  Set Master Password in Environment:")
    print(f"   export ATOM_MASTER_PASSWORD='{master_password}'")
    
    print("\n2️⃣  Make it Persistent (add to ~/.zshrc or ~/.bash_profile):")
    print(f"   echo \"export ATOM_MASTER_PASSWORD='{master_password}'\" >> ~/.zshrc")
    print("   source ~/.zshrc")
    
    print("\n3️⃣  Or Store in macOS Keychain (recommended):")
    print(f"   security add-generic-password -a $USER -s ATOM_MASTER_PASSWORD -w '{master_password}'")
    
    print("\n4️⃣  Verify Storage Location:")
    creds_dir = Path.home() / ".atom" / "credentials"
    print(f"   ls -la {creds_dir}/")
    
    print("\n5️⃣  Update main.py:")
    print("   - Replace hardcoded API key loading")
    print("   - Use: from core.secure_credentials import CredentialManager")
    print("   - Example: cm = CredentialManager(); key = cm.get('gemini_fast')")
    
    print("\n6️⃣  Test ATOM:")
    print("   python main.py")
    
    print("\n" + "="*70)
    print("✅ Setup Complete!")
    print("="*70)


def main():
    """Main setup routine."""
    print("\n" + "🔐 "*20)
    print("ATOM Secure API Key Setup")
    print("🔐 "*20)
    
    try:
        # Step 1: Master password
        master_password = setup_master_password()
        
        # Step 2: API keys
        if not setup_api_keys(master_password):
            print_error("Failed to set up API keys")
            sys.exit(1)
        
        # Step 3: Verification
        if not verify_setup(master_password):
            print_warning("Verification did not complete successfully")
            sys.exit(1)
        
        # Show next steps
        show_next_steps(master_password)
        
        print_success("All steps completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
