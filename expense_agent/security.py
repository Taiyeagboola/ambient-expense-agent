# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def scrub_description(
    description: str, already_redacted: list[str] | None = None
) -> tuple[str, list[str]]:
    """Detects and redacts SSNs and Credit Card numbers from the description.

    Returns a tuple of (scrubbed_description, redacted_categories_list).
    """
    if not description:
        return description, already_redacted or []

    redacted_categories = set(already_redacted or [])

    # Matches 13-16 digits with optional spaces or hyphens.
    cc_pattern = re.compile(r"\b(?:\d[ -]?){13,16}\b")

    # Matches XXX-XX-XXXX or 9 consecutive digits.
    ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")

    scrubbed = description

    if cc_pattern.search(scrubbed):
        scrubbed = cc_pattern.sub("[REDACTED_CC]", scrubbed)
        redacted_categories.add("Credit Card")

    if ssn_pattern.search(scrubbed):
        scrubbed = ssn_pattern.sub("[REDACTED_SSN]", scrubbed)
        redacted_categories.add("SSN")

    return scrubbed, sorted(redacted_categories)


def detect_prompt_injection(description: str) -> bool:
    """Checks if the description contains common prompt injection phrases."""
    if not description:
        return False

    patterns = [
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"ignore\s+(?:all\s+)?rules",
        r"bypass\s+rules",
        r"override\s+(?:all\s+)?instructions",
        r"force\s+auto-approve",
        r"force\s+approval",
        r"system\s+prompt",
        r"always\s+approve",
        r"auto-approve\s+this",
        r"auto\s+approve\s+this",
        r"you\s+must\s+approve",
        r"approve\s+this\s+expense",
        r"ignore\s+risk",
        r"bypass\s+risk",
    ]

    for pattern in patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return True

    return False
