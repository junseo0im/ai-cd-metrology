"""Process entry point: run ONE mapper-approved request through the unchanged BatchRunner.

Launched by the Dashboard in the model environment (torch/onnxruntime/cv2), which is
separate from the Streamlit environment. No SQLite access, no result adoption.

    python -m inspection.run_single_analysis --request request.json --output-root DIR

request.json: {"batch_id": str, "request": {...BatchRunner FIELDS...},
               "expected_sha256": str}
Prints one JSON line {"summary_json": path, "execution_status": ...} on success.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request = payload.get("request")
    batch_id = payload.get("batch_id")
    if not isinstance(request, dict) or isinstance(payload.get("requests"), list):
        raise SystemExit("exactly one request object is required")
    if request.get("batch_id") != batch_id:
        raise SystemExit("request batch_id mismatch")
    # Re-check the mapper-verified source bytes just before execution.
    with Path(request["image_path"]).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual.lower() != str(payload.get("expected_sha256", "")).lower():
        raise SystemExit("source image SHA-256 changed since input validation")
    from inspection.batch_runner import BatchRunner
    summary = BatchRunner().run(batch_id, [request], args.output_root)
    print(json.dumps({"summary_json": summary.get("summary_json"),
                      "execution_status": summary.get("execution_status")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
