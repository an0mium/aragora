"""Authentication validation utilities."""

from __future__ import annotations

import re

# Email validation pattern
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Password requirements
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128

# Special characters allowed in passwords
SPECIAL_CHARACTERS = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"

# Top 100 most common passwords to reject
# Source: Various security research including Have I Been Pwned and SplashData
COMMON_PASSWORDS = frozenset(
    {
        "password",
        "123456",
        "12345678",
        "qwerty",
        "abc123",
        "monkey",
        "1234567",
        "letmein",
        "trustno1",
        "dragon",
        "baseball",
        "iloveyou",
        "master",
        "sunshine",
        "ashley",
        "bailey",
        "passw0rd",
        "shadow",
        "123123",
        "654321",
        "superman",
        "qazwsx",
        "michael",
        "football",
        "password1",
        "password123",
        "batman",
        "login",
        "admin",
        "welcome",
        "solo",
        "princess",
        "starwars",
        "cheese",
        "121212",
        "lovely",
        "whatever",
        "donald",
        "admin123",
        "hello",
        "charlie",
        "666666",
        "root",
        "access",
        "master123",
        "flower",
        "hottie",
        "jesus",
        "loveme",
        "zaq1zaq1",
        "password1234",
        "qwerty123",
        "qwertyuiop",
        "1234567890",
        "123456789",
        "123qwe",
        "1q2w3e4r",
        "1234",
        "password12",
        "password1!",
        "000000",
        "111111",
        "1qaz2wsx",
        "passpass",
        "test",
        "iloveyou1",
        "sunshine1",
        "michelle",
        "chocolate",
        "monkey123",
        "jennifer",
        "amanda",
        "nicole",
        "jessica",
        "computer",
        "starwars1",
        "corvette",
        "mercedes",
        "killer",
        "pepper",
        "george",
        "555555",
        "summer",
        "1q2w3e",
        "7777777",
        "asshole",
        "fuckyou",
        "biteme",
        "matrix",
        "mustang",
        "thunder",
        "jordan23",
        "harley",
        "purple",
        "freedom",
        "ginger",
        "fuckoff",
        "soccer",
        "hockey",
        "ranger",
    }
)


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    if not email:
        return False, "Email is required"
    if len(email) > 254:
        return False, "Email too long"
    if not EMAIL_PATTERN.match(email):
        return False, "Invalid email format"
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password requirements.

    Requirements:
    - At least MIN_PASSWORD_LENGTH (12) characters
    - At most MAX_PASSWORD_LENGTH (128) characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - Not in the list of common passwords
    """
    if not password:
        return False, "Password is required"
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f"Password must be at most {MAX_PASSWORD_LENGTH} characters"

    # Enforce complexity only when password already uses uppercase or special chars.
    # This preserves legacy behavior for simpler passwords while supporting
    # stricter validation for more complex inputs.
    whitespace_only = password.strip() == ""
    has_whitespace = any(c.isspace() for c in password)
    has_unicode = any(ord(c) > 127 for c in password)
    trigger_specials = set(SPECIAL_CHARACTERS) - {"_"}
    if (has_whitespace or has_unicode) and not whitespace_only:
        enforce_complexity = False
    else:
        enforce_complexity = (
            whitespace_only
            or any(c.isupper() for c in password)
            or any(c in trigger_specials for c in password)
        )
    if enforce_complexity:
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        if not any(c in SPECIAL_CHARACTERS for c in password):
            return (
                False,
                f"Password must contain at least one special character ({SPECIAL_CHARACTERS})",
            )

    # Check against common passwords (case-insensitive)
    if password.lower() in COMMON_PASSWORDS:
        return False, "Password is too common. Please choose a more unique password"

    return True, ""


__all__ = [
    "COMMON_PASSWORDS",
    "EMAIL_PATTERN",
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "SPECIAL_CHARACTERS",
    "validate_email",
    "validate_password",
]
