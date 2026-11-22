"""Deploy Aegis to a free Hugging Face Docker Space (CPU basic: 2 vCPU, 16 GB RAM, $0).

Creates the Space if needed, stores secrets from your local environment as Space
secrets, and uploads the committed code (HEAD) with the Space README header.

PowerShell:
    pip install huggingface_hub
    $env:HF_TOKEN = "hf_..."                 # write token: https://huggingface.co/settings/tokens
    $env:AEGIS_ADMIN_PASSWORD = "..."        # dashboard login for user "admin"
    $env:GEMINI_API_KEY = "..."              # free key: https://aistudio.google.com/apikey
    python deploy/deploy_hf_space.py --space <hf-username>/aegis

Optional env vars forwarded as secrets: AEGIS_DEMO_PASSWORD (creates a shareable "demo"
operator), GEMINI_MODEL, RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET (test mode; mock links
otherwise), RAZORPAY_WEBHOOK_SECRET. JWT_SECRET_KEY is generated if you don't set one.
Use --dry-run to stage the upload locally without contacting Hugging Face.
"""
import argparse
import io
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPACE_HEADER = """---
title: Aegis
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Governed AI commerce agent - LLM proposes, policy decides
---

"""

# Binary artifacts must go through LFS/Xet on the Hub; keep the repo's own eol rules too.
SPACE_GITATTRIBUTES = """*.pkl filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.sh text eol=lf
Dockerfile text eol=lf
"""

FORWARDED_SECRETS = [
    "AEGIS_ADMIN_USERNAME", "AEGIS_ADMIN_PASSWORD", "AEGIS_DEMO_USERNAME", "AEGIS_DEMO_PASSWORD",
    "GEMINI_API_KEY", "GEMINI_MODEL", "JWT_SECRET_KEY",
    "RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_ENV", "RAZORPAY_WEBHOOK_SECRET",
    "AEGIS_PUBLIC_UI",  # "dashboard" (default) or "console"
]
# Not needed at runtime; keeps the Space small.
EXCLUDE_PREFIXES = ("notebooks-", ".github/")


def stage(dest: str) -> str:
    """Export HEAD (committed files only — never local .env or databases) into dest."""
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    archive = subprocess.check_output(["git", "-C", ROOT, "archive", "--format=tar", "HEAD"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = [m for m in tar.getmembers() if not m.name.startswith(EXCLUDE_PREFIXES)]
        tar.extractall(dest, members=members)
    readme = os.path.join(dest, "README.md")
    with open(readme, encoding="utf-8") as f:
        body = f.read()
    with open(readme, "w", encoding="utf-8", newline="\n") as f:
        f.write(SPACE_HEADER + body)
    with open(os.path.join(dest, ".gitattributes"), "w", encoding="utf-8", newline="\n") as f:
        f.write(SPACE_GITATTRIBUTES)
    return sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true", help="stage files locally and stop")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="aegis-space-")
    try:
        sha = stage(tmp)
        files = sorted(os.path.relpath(os.path.join(d, f), tmp).replace("\\", "/")
                       for d, _, fs in os.walk(tmp) for f in fs)
        print(f"Staged {len(files)} files from commit {sha}")
        if args.dry_run:
            print("\n".join(files))
            return

        from huggingface_hub import HfApi  # pip install huggingface_hub
        api = HfApi(token=os.getenv("HF_TOKEN") or None)
        print(f"Authenticated to Hugging Face as {api.whoami()['name']}")

        if not os.getenv("AEGIS_ADMIN_PASSWORD"):
            print("WARNING: AEGIS_ADMIN_PASSWORD not set - nobody will be able to sign in to the dashboard.")
        if not os.getenv("GEMINI_API_KEY"):
            print("WARNING: GEMINI_API_KEY not set - agent runs will return 503 (no LLM reachable).")

        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        values = {k: os.environ[k] for k in FORWARDED_SECRETS if os.getenv(k)}
        values.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))
        for key, value in values.items():
            api.add_space_secret(args.space, key, value)
        print(f"Set {len(values)} Space secrets: {', '.join(sorted(values))}")

        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message=f"Deploy {sha}", delete_patterns=["*"])
        owner, name = args.space.split("/", 1)
        print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
        print(f"App URL (after the build, ~5 min): https://{owner}-{name}.hf.space".lower().replace("_", "-"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
