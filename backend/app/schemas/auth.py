"""
RadAssist AI — Auth Schemas (Phase 6)

Registration exists and is gated by ALLOW_REGISTRATION.

⚠️  IT IS SAFE BECAUSE OF OWNERSHIP, NOT INSTEAD OF IT.
Open signup was rejected while every signed-in user could see every report.
Once reports and images carry an owner (migration 0006), a new account lands
in an empty workspace and the objection no longer applies. If ownership were
ever reverted, ALLOW_REGISTRATION would have to go false in the same change.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    # ⚠️  NO max_length THAT WOULD REVEAL THE POLICY, AND NO min_length.
    # Validation limits on a LOGIN form tell an attacker the password rules
    # before they have an account. Length is enforced where passwords are
    # SET (the create_user script), which is the only place it protects
    # anything. Here, a wrong-length password is simply a wrong password.
    password: str = Field(repr=False)

    # repr=False above, and this, keep the password out of validation errors
    # and out of any log line that formats the model.
    model_config = ConfigDict(extra="forbid")


class RegisterRequest(BaseModel):
    """
    Create an account.

    ⚠️  NO is_admin FIELD, AND THERE MUST NEVER BE ONE.
    A client-settable admin flag is a privilege escalation with a form
    attached. Administrators are made by an operator running
    scripts/create_user.py --admin, which requires shell access to the
    container.
    """
    email: EmailStr
    # min_length HERE, not on LoginRequest. A length rule where a password is
    # SET protects the account; the same rule on the login form only tells an
    # attacker the policy before they have one.
    password: str = Field(min_length=12, max_length=72, repr=False)
    full_name: str | None = Field(default=None, max_length=200)

    model_config = ConfigDict(extra="forbid")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")


class UserResponse(BaseModel):
    """
    A user, as returned to that user.

    Carries no hash, no admin flag decisions and nothing about other accounts.
    """
    id: UUID
    email: str
    full_name: str | None = None
    is_admin: bool
    last_login_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
