"""Process entry point: run mapper-approved READY requests, one unchanged BatchRunner run per Batch.

BatchRunner.run accepts exactly one batch_id, so a multi-Batch selection (e.g. a Wafer)
is executed as sequential, separate BatchRunner runs - Batches are never merged. Each run
writes its own runs/batch-<uuid>/ folder. No SQLite access, no adoption.

    python -m inspection.run_dashboard_analysis --request request.json --output-root DIR

request.json: {"groups": [{"batch_id": str, "requests": [{...}, ...]}, ...],
               "expected_sha256": {image_id: sha256}, "confirmed_count": int|null}

More than CONFIRM_THRESHOLD images run only when confirmed_count equals the exact
number of requested images (the Dashboard's explicit large-run confirmation).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

CONFIRM_THRESHOLD = 3  # Larger runs need an explicit, exact-count confirmation.


def validate(payload):
    groups = payload.get("groups")
    expected = payload.get("expected_sha256")
    if not isinstance(groups, list) or not groups or not isinstance(expected, dict):
        raise ValueError("groups and expected_sha256 are required")
    seen = set()
    for group in groups:
        batch_id = group.get("batch_id")
        requests = group.get("requests")
        if not isinstance(batch_id, str) or not batch_id or not isinstance(requests, list) or not requests:
            raise ValueError("each group needs a batch_id and a nonempty request list")
        for request in requests:
            image_id = request.get("image_id")
            if request.get("batch_id") != batch_id:
                raise ValueError("request batch_id differs from its group")
            if image_id in seen:
                raise ValueError("duplicate image_id across the selection")
            seen.add(image_id)
    confirmed = payload.get("confirmed_count")
    if len(seen) > CONFIRM_THRESHOLD and (isinstance(confirmed, bool) or confirmed != len(seen)):
        raise ValueError(f"{len(seen)} images require an explicit large-run confirmation of exactly that count")
    if set(expected) != seen:
        raise ValueError("expected_sha256 must cover exactly the requested images")
    # Re-check every mapper-verified source before any model is loaded.
    for group in groups:
        for request in group["requests"]:
            with Path(request["image_path"]).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual.lower() != str(expected[request["image_id"]]).lower():
                raise ValueError("source image SHA-256 changed since input validation: " + request["image_id"])
    return groups


def run_groups(payload, output_root, runner):
    """Validate everything first, then run each Batch group sequentially."""
    groups = validate(payload)
    results = []
    for group in groups:
        try:
            summary = runner.run(group["batch_id"], group["requests"], output_root)
            results.append({"batch_id": group["batch_id"], "summary_json": summary.get("summary_json"),
                            "execution_status": summary.get("execution_status")})
        except Exception as error:  # One Batch's structural failure does not stop the others.
            results.append({"batch_id": group["batch_id"], "summary_json": None,
                            "execution_status": "FAILED", "error": f"{type(error).__name__}: {error}"})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        validate(payload)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
    from inspection.batch_runner import BatchRunner
    runner = BatchRunner()  # One runner instance: models are loaded once and reused.
    print(json.dumps({"groups": run_groups(payload, args.output_root, runner)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
