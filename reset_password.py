# reset_password.py

import os
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reset a user's password.")
    parser.add_argument("--email", required=True, help="The email of the user to update.")
    parser.add_argument("--password", required=True, help="The new password for the user.")
    args = parser.parse_args()

    url: str = os.environ.get("SUPABASE_URL")
    service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase_admin: Client = create_client(url, service_key)
    print("Connecting to Supabase as admin...")

    try:
        # Get the list of all users
        all_users_list = supabase_admin.auth.admin.list_users()
        target_user = None
        
        for u in all_users_list:
            if u.email == args.email:
                target_user = u
                break
        
        if not target_user:
            raise Exception(f"User with email '{args.email}' not found.")

        print(f"Found user. Attempting to update password for user ID: {target_user.id}")

        # --- THIS IS THE FIX ---
        # The user ID is the first POSITIONAL argument.
        # 'attributes' is a KEYWORD argument.
        supabase_admin.auth.admin.update_user_by_id(
            target_user.id,  # Correct
            attributes={'password': args.password}
        )
        # --- END OF FIX ---
        
        print("\n✅ --- Success! --- ✅")
        print(f"Password for user '{args.email}' has been successfully reset.")

    except Exception as e:
        print("\n❌ --- An Error Occurred --- ❌")
        print(e)

if __name__ == "__main__":
    main()