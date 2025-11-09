#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Optional, Dict

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

MLRUNS_ROOT = os.environ.get("MLRUNS_ROOT", "Workflow-CI/MLProject/mlruns")
DEST_PARENT_ID = os.environ["GDRIVE_FOLDER_ID"]  # folder/drive tujuan
creds_json = json.loads(os.environ["GDRIVE_CREDENTIALS"])
credentials = Credentials.from_service_account_info(
    creds_json, scopes=["https://www.googleapis.com/auth/drive"]
)
service = build("drive", "v3", credentials=credentials, cache_discovery=False)

def find_child_by_name(parent_id: str, name: str) -> Optional[Dict]:
    """Cari item bernama 'name' di dalam parent_id (Shared Drive aware)."""
    q = "name = %(name)r and '%(parent)s' in parents and trashed = false" % {
        "name": name, "parent": parent_id
    }
    resp = service.files().list(
        q=q,
        spaces="drive",
        fields="files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None

def ensure_folder(parent_id: str, name: str) -> str:
    """Pastikan folder dengan nama 'name' ada di parent_id. Return folder_id."""
    existing = find_child_by_name(parent_id, name)
    if existing and existing.get("mimeType") == "application/vnd.google-apps.folder":
        return existing["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = service.files().create(
        body=meta, fields="id", supportsAllDrives=True
    ).execute()
    return created["id"]

def upload_file(local_path: Path, parent_id: str):
    media = MediaFileUpload(local_path.as_posix(), resumable=True)
    body = {"name": local_path.name, "parents": [parent_id]}
    service.files().create(
        body=body,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()

def upload_directory(local_dir: Path, parent_id: str):
    for item in sorted(local_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            folder_id = ensure_folder(parent_id, item.name)
            upload_directory(item, folder_id)
        else:
            print(f"Uploading file: {item}")
            upload_file(item, parent_id)

def main():
    mlruns_path = Path(MLRUNS_ROOT)
    if not mlruns_path.exists():
        raise SystemExit(f"Path not found: {mlruns_path}")

    # Struktur: mlruns/<experiment_id>/<run_id>/
    exp_dirs = [p for p in mlruns_path.iterdir() if p.is_dir() and not p.name.startswith(".")]

    if not exp_dirs:
        print("No experiments found under mlruns/. Nothing to upload.")
        return

    for exp_dir in exp_dirs:
        run_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
        print(f"[INFO] Experiment {exp_dir.name}: {len(run_dirs)} run(s)")

        # Buat folder experiment di Drive (opsional, biar rapi)
        exp_drive_id = ensure_folder(DEST_PARENT_ID, exp_dir.name)

        for run_dir in run_dirs:
            # Buat folder run_id di bawah experiment
            run_drive_id = ensure_folder(exp_drive_id, run_dir.name)
            print(f"=== Uploading run: {exp_dir.name}/{run_dir.name} ===")
            upload_directory(run_dir, run_drive_id)

    print("=== All experiments/runs uploaded to Google Drive ===")

if __name__ == "__main__":
    try:
        main()
    except HttpError as e:
        print(f"[DriveAPI] HttpError: {e}")
        raise