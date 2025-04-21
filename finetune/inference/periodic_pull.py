#!/usr/bin/env python3
"""
Periodically upload a local file (already inside this VM) to Google Drive.

Example run *inside the VM*:
    python3 periodic_pull.py \
        --file arc_problems_validation_400_vllm_generated.jsonl \
        --gdrive-folder-id 1dN8AbCdEfGhIjKlMn0PqRsTuVwXyZ \
        --interval 120
"""

import argparse, time, sys
from datetime import datetime
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def drive_service():
    creds = None
    if Path("token.json").exists():
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        Path("token.json").write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)

def find_file(svc, name, folder):
    q = f"name='{name}' and '{folder}' in parents and trashed=false"
    res = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    return res[0]["id"] if res else None

def upload(svc, local, folder):
    fname = Path(local).name
    fid   = find_file(svc, fname, folder)
    media = MediaFileUpload(local, resumable=True)
    if fid:
        svc.files().update(fileId=fid, media_body=media).execute()
    else:
        meta = {"name": fname, "parents": [folder]}
        fid = svc.files().create(body=meta, media_body=media, fields="id").execute()["id"]
    return fid

def parse():
    p = argparse.ArgumentParser("Local‑file → Google Drive uploader (runs inside VM)")
    p.add_argument("--file", required=True, help="Path to the file inside this VM")
    p.add_argument("--gdrive-folder-id", required=True, help="Destination Drive folder ID")
    p.add_argument("--interval", type=int, default=300, help="Polling interval in seconds")
    return p.parse_args()

def main():
    args  = parse()
    path  = Path(args.file).expanduser().resolve()
    last  = 0
    svc   = drive_service()
    print(f"📂 Watching: {path}")
    while True:
        if path.exists():
            m = path.stat().st_mtime
            if m > last:
                fid = upload(svc, path, args.gdrive_folder_id)
                ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"✅ {ts} – uploaded → https://drive.google.com/file/d/{fid}")
                last = m
        else:
            print("Waiting for the file to appear …")
        time.sleep(args.interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr); sys.exit(1)
