import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.validate_change_metadata import validate_commit, validate_pr


GOOD_COMMIT = """perf(v8/bf16): accelerate vision projections

Why: Qwen3-VL projectors dominated encoder time.
What: Add an AMX provider with fail-closed execution contracts.
Validation: BF16 oracle 4/4; v8 regression passed.
Evidence: Encoder 270.8s to 28.1s; report path build/report.json.
Docs: docs/site/_pages/bf16-amx-performance-study.html
Nightly: test-bf16-performance-sweep reports PASS or hardware SKIP.
Content: Angle=AMX optimization under numerical contracts; Claims=9.6x encoder progression; Caveats=managed Xeon, not bare metal; Sources=study page, JSON report, commit diff
"""

GOOD_PR = """## Why
Projectors dominated encoder time.
## What changed
Added an explicit AMX provider.
## Evidence
Encoder moved from 270.8s to 28.1s.
## Validation
BF16 oracle and v8 regression passed.
## Regression coverage
Nightly reports PASS or hardware SKIP.
## Documentation
docs/site/_pages/bf16-amx-performance-study.html
## Content handoff
- Audience: CPU AI engineers
- Angle: AMX speed without relaxing numerical contracts
- Claims: 9.6x measured encoder progression
- Caveats: managed Xeon allocation, not bare-metal limits
- Sources: study page, generated JSON report, commit diff
"""


class ChangeMetadataTests(unittest.TestCase):
    def test_substantive_commit_requires_complete_handoff(self):
        self.assertEqual(validate_commit(GOOD_COMMIT), [])
        errors = validate_commit(GOOD_COMMIT.replace("Evidence: Encoder", "Evidence: <result>\nIgnored: Encoder"))
        self.assertTrue(any("Evidence" in error for error in errors))

    def test_non_publishable_commit_requires_reason(self):
        message = GOOD_COMMIT.replace(
            "Content: Angle=AMX optimization under numerical contracts; Claims=9.6x encoder progression; Caveats=managed Xeon, not bare metal; Sources=study page, JSON report, commit diff",
            "Content: not publishable; internal path normalization with no user-visible behavior",
        )
        self.assertEqual(validate_commit(message), [])

    def test_pr_requires_content_fields(self):
        self.assertEqual(validate_pr(GOOD_PR), [])
        errors = validate_pr(GOOD_PR.replace("- Caveats:", "- Limitations:"))
        self.assertIn("Content handoff must include Caveats:", errors)

    def test_pr_can_explain_why_not_publishable(self):
        body = GOOD_PR.rsplit("## Content handoff", 1)[0] + (
            "## Content handoff\nNot publishable: mechanical test fixture rename only\n"
        )
        self.assertEqual(validate_pr(body), [])

    def test_template_comments_do_not_satisfy_sections(self):
        body = "## Why\n<!-- fill this in -->\n## What changed\n<!-- fill this in -->"
        errors = validate_pr(body)
        self.assertIn("missing or empty PR section: Why", errors)


class PullRequestTemplateTests(unittest.TestCase):
    def test_checked_in_pr_template_covers_all_required_sections(self):
        # Ratchet: if PR_SECTIONS changes, the template must change with it.
        import re
        from pathlib import Path

        from scripts.validate_change_metadata import PR_SECTIONS

        template = (
            Path(__file__).resolve().parents[1] / ".github" / "pull_request_template.md"
        ).read_text(encoding="utf-8")
        headings = {
            match.group(1).strip().lower()
            for match in re.finditer(r"(?m)^##\s+(.+?)\s*$", template)
        }
        missing = [name for name in PR_SECTIONS if name.lower() not in headings]
        self.assertEqual(missing, [])


class PrePushMetadataHookTests(unittest.TestCase):
    def test_deletion_only_push_does_not_validate_checked_out_head(self):
        root = Path(__file__).resolve().parents[1]
        zero = "0" * 40
        with tempfile.TemporaryDirectory(prefix="ck_prepush_delete_") as tmp:
            env = dict(os.environ)
            env.update(
                {
                    "CK_PREPUSH_ALLOW_LOW_RESOURCES": "1",
                    "CK_PREPUSH_LOCK_PATH": str(Path(tmp) / "prepush.lock"),
                }
            )
            completed = subprocess.run(
                [str(root / ".githooks" / "pre-push"), "origin", "unused"],
                cwd=root,
                env=env,
                input=f"(delete) {zero} refs/heads/obsolete {'1' * 40}\n",
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Deletion-only push", completed.stdout)
        self.assertNotIn("Validating commit change metadata", completed.stdout)

    def test_metadata_log_is_unique_and_cleaned_up(self):
        hook = (Path(__file__).resolve().parents[1] / ".githooks" / "pre-push").read_text(
            encoding="utf-8"
        )
        self.assertIn('CHANGE_METADATA_LOG="$(mktemp ', hook)
        self.assertNotIn("/tmp/ck_change_metadata.log", hook)
        self.assertIn('rm -f "$CHANGE_METADATA_LOG"', hook)


if __name__ == "__main__":
    unittest.main()
