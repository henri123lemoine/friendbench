import json
import zipfile

files = {
    # TBD
}


def load_samples(path):
    samples = []
    with zipfile.ZipFile(path, "r") as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with z.open(name) as f:
                    samples.append(json.load(f))
    samples.sort(key=lambda s: s.get("id", 0))
    return samples


for label, path in files.items():
    samples = load_samples(path)

    print(f"\n{'=' * 80}")
    print(f"MODEL: {label}")
    print(f"{'=' * 80}")

    for sample in samples:
        messages = sample.get("messages", [])
        input_text = ""
        response_text = ""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if msg.get("role") == "user" and not input_text:
                input_text = content
            if msg.get("role") == "assistant":
                response_text = content

        scores = sample.get("scores", {})
        score_info = scores.get("model_graded_qa", {})
        score_val = score_info.get("value", "N/A")
        explanation = score_info.get("explanation", "N/A")

        marker = "PASS" if score_val == "C" else "FAIL"
        print(f"\n--- Sample {sample.get('id', '?')} | {marker} ---")
        print(f"Q: {input_text}")
        print(f"\nRESPONSE:\n{response_text[:600]}")
        print(f"\nGRADER REASONING:\n{explanation[:600]}")
        print()
