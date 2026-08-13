#!/usr/bin/env python
"""
Create a RadAssist account. Run by whoever operates the deployment.

    docker-compose exec backend python scripts/create_user.py \
        --email radiologist@hospital.org --name "Dr A. Jones"

    # first account, needs to be able to create others:
    docker-compose exec backend python scripts/create_user.py \
        --email you@example.org --admin

════════════════════════════════════════════════════════════════════
⚠️  THIS SCRIPT EXISTS SO THAT /auth/register DOES NOT
════════════════════════════════════════════════════════════════════
A public registration endpoint on a clinical tool means anyone who finds the
URL can create an account and read uploaded patient reports. The pilot has
known users by name, so account creation belongs to the operator — an
attacker cannot reach a command that requires shell access to the container.

⚠️  THE PASSWORD IS NEVER TAKEN FROM THE COMMAND LINE.
`--password hunter2` would land in shell history, in `ps` output for every
other user on the machine, and in Docker's own logs. It is prompted for, with
echo off, or generated here and printed once.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import secrets
import sys
from pathlib import Path

# Run as a script from anywhere: `python scripts/create_user.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select                                   # noqa: E402

from app.core.database import async_session                     # noqa: E402
from app.core.security import AuthError, hash_password          # noqa: E402
from app.models.user import User                                # noqa: E402


def _read_password() -> tuple[str, bool]:
    """
    Prompt for a password, or generate one. Returns (password, was_generated).

    getpass keeps it off the screen and out of the terminal's scrollback.
    An empty prompt means "generate one for me", which is the better default
    for a clinical account — a generated 24-byte secret beats whatever
    somebody types at 7pm.
    """
    first = getpass.getpass("Password (leave blank to generate one): ")

    if not first:
        return secrets.token_urlsafe(18), True

    if len(first) < 12:
        print("✗ Too short. Use at least 12 characters.", file=sys.stderr)
        sys.exit(1)

    second = getpass.getpass("Confirm: ")
    if first != second:
        print("✗ Passwords did not match.", file=sys.stderr)
        sys.exit(1)

    return first, False


async def create(email: str, name: str | None, admin: bool) -> int:
    email = User.normalise_email(email)
    if "@" not in email:
        print(f"✗ {email!r} is not an email address.", file=sys.stderr)
        return 1

    async with async_session() as db:
        existing = (await db.execute(
            select(User).where(User.email == email)
        )).scalar_one_or_none()

        if existing:
            # Not an update path on purpose: silently resetting an existing
            # clinician's password because someone re-ran the command with the
            # same address would be a surprising way to lock them out.
            print(f"✗ {email} already exists.", file=sys.stderr)
            return 1

        password, generated = _read_password()

        try:
            hashed = hash_password(password)
        except AuthError as e:
            print(f"✗ {e}", file=sys.stderr)
            return 1

        db.add(User(
            email=email,
            hashed_password=hashed,
            full_name=name,
            is_admin=admin,
            is_active=True,
        ))
        await db.commit()

    print(f"\n✅ Created {email}" + ("  (administrator)" if admin else ""))
    if generated:
        # Printed once, never stored anywhere but the bcrypt hash. There is
        # no reset flow — if this is lost, the operator deletes the row and
        # runs the script again.
        print(f"   Password: {password}")
        print("   Shown once. Give it to the user over a secure channel.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a RadAssist account (no public registration exists)."
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default=None, help="Display name.")
    parser.add_argument(
        "--admin", action="store_true",
        help="May create other accounts. Grant to the operator only.",
    )
    # ⚠️  There is deliberately no --password flag. See the module docstring.
    args = parser.parse_args()

    return asyncio.run(create(args.email, args.name, args.admin))


if __name__ == "__main__":
    raise SystemExit(main())
