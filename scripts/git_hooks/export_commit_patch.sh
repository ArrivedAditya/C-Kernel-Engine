#!/usr/bin/env bash
set -u

commit="${1:-HEAD}"
repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
git_dir="$(git rev-parse --git-dir 2>/dev/null)" || exit 0
commit_sha="$(git rev-parse --verify "${commit}^{commit}" 2>/dev/null)" || exit 0
short_sha="$(git rev-parse --short=12 "$commit_sha")" || exit 0
subject="$(git log -1 --format=%s "$commit_sha")" || exit 0
safe_subject="$(
    printf '%s' "$subject" |
        tr '[:upper:]' '[:lower:]' |
        sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/^(.{1,64}).*$/\1/'
)"
[ -n "$safe_subject" ] || safe_subject="commit"

archive_root="${CK_PATCH_ARCHIVE_DIR:-$HOME/.cache/ck-engine-v8/patches}"
timestamp="$(git log -1 --date=format-local:%Y%m%d-%H%M%S --format=%cd "$commit_sha")"
patch_path="$archive_root/${timestamp}-${short_sha}-${safe_subject}.patch"
sha_path="${patch_path}.sha256"

mkdir -p "$archive_root" || exit 0
tmp_patch="$(mktemp "$archive_root/.cke-patch.XXXXXX")" || exit 0
if ! git -C "$repo_root" format-patch --stdout -1 "$commit_sha" >"$tmp_patch"; then
    unlink "$tmp_patch"
    exit 0
fi
mv "$tmp_patch" "$patch_path" || exit 0
(
    cd "$archive_root" || exit 0
    sha256sum "$(basename "$patch_path")" >"$(basename "$sha_path")"
) || exit 0

printf 'CKE patch archive: %s\n' "$patch_path"
printf 'CKE patch SHA-256: %s\n' "$(cut -d' ' -f1 "$sha_path")"

# Keep hooks observational. A storage or tooling failure must never reject a commit.
exit 0
