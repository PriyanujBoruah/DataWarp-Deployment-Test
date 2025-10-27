import os
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv
import sys

# --- SCRIPT TO CREATE A NEW ORGANIZATION AND ITS FIRST ADMIN USER ---
# This script is intended for super-admin use only.

def main():
    """
    Creates a new organization, initializes its usage metrics, creates an
    admin user for it in Supabase Auth, and links the user to the
    organization in the public profiles table.
    """
    load_dotenv()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Create a new organization and admin user for the platform.")
    parser.add_argument("--org", required=True, help="The name of the new organization.")
    parser.add_argument("--email", required=True, help="The email for the organization's first admin user.")
    parser.add_argument("--password", required=True, help="The initial password for the admin user.")
    args = parser.parse_args()

    # --- Initialize Admin Client (uses the powerful SERVICE_ROLE key) ---
    url: str = os.environ.get("SUPABASE_URL")
    service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not url or not service_key:
        print("❌ ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file.")
        sys.exit(1)

    supabase_admin: Client = create_client(url, service_key)
    print("Connecting to Supabase as admin...")

    org_id = None
    user_id = None

    try:
        # --- Step 1: Create the Organization ---
        print(f"Creating organization: '{args.org}'...")
        org_res = supabase_admin.table('organizations').insert({"name": args.org}).execute()
        
        if not org_res.data:
            raise Exception("Failed to create organization.", getattr(org_res, 'error', 'Unknown error'))
        
        org_id = org_res.data[0]['id']
        print(f"✅ Successfully created organization with ID: {org_id}")

        # --- Step 2: Create the User in Supabase Auth ---
        print(f"Creating user: '{args.email}'...")
        user_res = supabase_admin.auth.admin.create_user({
            "email": args.email,
            "password": args.password,
            "email_confirm": True  # Auto-confirm the email since we are the admin
        })

        if not user_res.user:
            raise Exception("Failed to create user in Auth.", getattr(user_res, 'error', 'Unknown error'))

        user_id = user_res.user.id
        print(f"✅ Successfully created user with ID: {user_id}")

        # --- Step 3: Link the User to the Organization in the 'profiles' table ---
        print("Linking user to organization and assigning 'admin' role...")
        profile_res = supabase_admin.table('profiles').insert({
            "id": user_id,
            "organization_id": org_id,
            "role": "admin"  # The first user of an org is always an admin
        }).execute()
        
        if not profile_res.data:
            raise Exception("Failed to create user profile.", getattr(profile_res, 'error', 'Unknown error'))
        
        print("✅ Successfully created user profile.")

        # --- Step 4: Initialize Usage Metrics for the new Organization ---
        print("Initializing usage metrics...")
        metrics_res = supabase_admin.table('usage_metrics').insert({
            "organization_id": org_id,
            "user_count": 1,  # Start with the one admin user we just created
            "pipeline_count": 0,
            "rows_processed": 0
        }).execute()
        
        if not metrics_res.data:
            raise Exception("Failed to create usage_metrics row.", getattr(metrics_res, 'error', 'Unknown error'))

        print("✅ Successfully initialized usage metrics.")
        print("\n🎉 --- ALL STEPS COMPLETED SUCCESSFULLY --- 🎉")
        print(f"Organization '{args.org}' and admin user '{args.email}' are ready to use.")

    except Exception as e:
        print("\n❌ --- AN ERROR OCCURRED: ROLLING BACK CHANGES --- ❌")
        print(e)
        
        # This is a critical rollback process to prevent orphaned data.
        if user_id:
            print(f"Attempting to delete orphaned user: {user_id}")
            try:
                supabase_admin.auth.admin.delete_user(user_id)
                print("✅ Orphaned user deleted.")
            except Exception as delete_err:
                print(f"⚠️ Could not delete orphaned user. Manual cleanup may be required. Error: {delete_err}")

        if org_id:
            print(f"Attempting to delete orphaned organization: {org_id}")
            try:
                # The CASCADE DELETE on the foreign keys should handle deleting the profile and metrics
                supabase_admin.table('organizations').delete().eq('id', org_id).execute()
                print("✅ Orphaned organization deleted.")
            except Exception as delete_err:
                print(f"⚠️ Could not delete orphaned organization. Manual cleanup may be required. Error: {delete_err}")
        
        sys.exit(1) # Exit with an error code

if __name__ == "__main__":
    main()