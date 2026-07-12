import unittest

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


if __name__ == "__main__":
    unittest.main()
