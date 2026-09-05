"""Recorded episodes -> chat-format SFT JSONL.

Usage: python tools/make_sft.py jobs/<job>/episodes --out sft.jsonl [--relabel oracle|none]
                                [--only-failures] [--include-human]
Failed steps are relabeled with what the oracle would have done from that exact state (DAgger style);
human demonstrations are their own labels.
"""
import sys, os, json, glob, gzip, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy import SKILL_SYSTEM, oracle


def build(episodes_dir, out_path, relabel="oracle", only_failures=False, include_human=True):
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as out:
        # Closed episodes are rotated into episodes/archive/*.jsonl.gz by runtime_ops, so read both.
        # Globbing only *.jsonl silently drops every archived episode from the training data.
        paths = sorted(glob.glob(os.path.join(episodes_dir, "*.jsonl")))
        paths += sorted(glob.glob(os.path.join(episodes_dir, "archive", "*.jsonl.gz")))
        for path in paths:
            try:
                opener = gzip.open if path.endswith(".gz") else open
                with opener(path, "rt") as f:
                    lines = [json.loads(l) for l in f if l.strip()]
            except Exception:
                continue
            if not lines:
                continue
            head = lines[0]
            foot = next((l for l in lines if l.get("type") == "footer"), None)
            if foot is None:
                continue
            human = ((head.get("cfg") or {}).get("policy") or {}).get("kind") == "human"
            if human and not include_human:
                continue
            if only_failures and foot.get("success") and not human:
                continue
            for st in (l for l in lines if l.get("type") == "step"):
                obs = st.get("obs") or {}
                if not obs:
                    continue
                if human or relabel == "none":
                    label = st["action"]
                else:
                    try:
                        label = oracle(obs)
                    except Exception:
                        label = st["action"]
                    if st["result"].get("ok") is False and label == st["action"]:
                        continue
                label = {k: v for k, v in label.items() if k != "reasoning" and v is not None}
                out.write(json.dumps({
                    "messages": [{"role": "system", "content": SKILL_SYSTEM},
                                 {"role": "user", "content": json.dumps(obs)},
                                 {"role": "assistant", "content": json.dumps(label)}],
                    "meta": {"episode_id": head.get("episode_id"), "step": st.get("step"),
                             "was_failure": not st["result"].get("ok"), "tags": foot.get("tags"),
                             "source": "human" if human else ((head.get("cfg") or {}).get("policy") or {}).get("kind")}
                }) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes_dir")
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--relabel", default="oracle", choices=["oracle", "none"])
    ap.add_argument("--only-failures", action="store_true")
    ap.add_argument("--include-human", action="store_true")
    a = ap.parse_args()
    n = build(a.episodes_dir, a.out, a.relabel, a.only_failures, a.include_human or True)
    print(f"wrote {n} examples to {a.out}")
