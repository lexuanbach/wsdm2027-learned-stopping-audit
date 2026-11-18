"""Predeclared semantic-verification feature probe for guarded FBG stopping.

The tenth feature is entailment probability from the fixed
cross-encoder/nli-MiniLM2-L6-H768 checkpoint (commit b95119c) between
cumulative evidence and ``The answer to <question> is <answer>.``  The critic,
cross-fitting, threshold grid, 1-EM training band, and answer-repeat guard are
otherwise unchanged.  Seed 7 is exploratory (designed after a partial look);
fresh seed/sample replications use the frozen procedure.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import run_experiment as rx


MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
GRID = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]


def inputs(rows):
    keys, premises, hypotheses = [], [], []
    for split, row in rows:
        for t, prefix in enumerate(row["trajectory"]):
            keys.append((split, str(row["id"]), t))
            premises.append(" ".join(prefix["evidence"]) or "No retrieved evidence.")
            hypotheses.append(
                f"The answer to the question '{row['question']}' is "
                f"'{prefix['answer']}'.")
    return keys, premises, hypotheses


def entailment_scores(rows, batch_size=32, device_name="auto"):
    keys, premises, hypotheses = inputs(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                               local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True)
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device).eval()
    label = next(i for i, name in model.config.id2label.items()
                 if str(name).lower() == "entailment")
    values = []
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            batch = tokenizer(premises[start:start + batch_size],
                              hypotheses[start:start + batch_size],
                              padding=True, truncation=True, max_length=384,
                              return_tensors="pt")
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            logits = model(**batch).logits
            values.extend(torch.softmax(logits, dim=-1)[:, label]
                          .detach().cpu().numpy().tolist())
    return dict(zip(keys, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="corpus_guarded_qwen3b")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run = json.loads((MEAS / f"run_{args.tag}.json").read_text())
    cache = pickle.loads((MEAS / f"critic_data_{args.tag}.pkl").read_bytes())
    rows = ([('train', row) for row in cache['traces']] +
            [('test', row) for row in run['per_q']])
    feature_path = MEAS / f"semantic_features_{args.tag}.json"
    if feature_path.exists():
        semantic = {(r["split"], r["id"], r["t"]): r["entailment"]
                    for r in json.loads(feature_path.read_text())["rows"]}
    else:
        semantic = entailment_scores(rows, batch_size=args.batch_size,
                                     device_name=args.device)
        feature_path.write_text(json.dumps({
            "model": MODEL, "revision": REVISION,
            "definition": "NLI entailment(cumulative evidence, answer-to-question hypothesis)",
            "rows": [{"split": key[0], "id": key[1], "t": key[2],
                      "entailment": value} for key, value in semantic.items()],
        }, indent=1) + "\n")

    X = np.asarray(cache["X_val"], float)
    y_val = np.asarray(cache["y_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    train_semantic = []
    for row in cache["traces"]:
        train_semantic.extend(semantic[("train", str(row["id"]), t)]
                              for t in range(len(row["trajectory"])))
    X_aug = np.column_stack([X, np.asarray(train_semantic)])
    all_chains = rx.chain_indices(X_aug)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains)); rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    chains = [all_chains[i] for i in order[:-ncal]]
    oof, _ = rx.crossfit_critic_scores(X_aug, y_val, y_err, chains)
    threshold, selection = rx.select_frozen_lambda(
        oof, X_aug, y_err, chains, GRID, "guarded")
    fit_rows = [i for chain in chains for i in chain]
    critic = rx.VoICritic()
    critic.fit(X_aug[fit_rows], y_err[fit_rows],
               X_aug[fit_rows], y_val[fit_rows])

    per_q = []
    for row in run["per_q"]:
        scores = []
        for t, base in enumerate(row["prefix_features"]):
            augmented = list(base) + [semantic[("test", str(row["id"]), t)]]
            scores.append(critic.value(augmented))
        value_stop = next((t for t, value in enumerate(scores[:-1])
                           if value < threshold), len(scores) - 1)
        repeat_stop = len(scores) - 1
        for t in range(1, len(scores)):
            if rx._norm(row["trajectory"][t]["answer"]) == rx._norm(
                    row["trajectory"][t - 1]["answer"]):
                repeat_stop = t
                break
        stop = max(value_stop, repeat_stop)
        per_q.append({"id": str(row["id"]), "stop": stop,
                      "em": row["prefix_em"][stop], "scores": scores})

    status = ("exploratory_family_defined_after_partial_seed7_look"
              if args.tag == "corpus_guarded_qwen3b"
              else "predeclared_fresh_replication")
    result = {
        "method": "semantic_entailment_augmented_guarded_fbg",
        "status": status, "tag": args.tag,
        "nli_model": MODEL, "nli_revision": REVISION,
        "threshold": threshold, "selection": selection,
        "per_q": per_q,
    }
    dest = MEAS / f"semantic_guard_{args.tag}.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": status, "threshold": threshold,
                      "n": len(per_q),
                      "em": 100 * float(np.mean([r["em"] for r in per_q])),
                      "searches": float(np.mean([r["stop"] for r in per_q]))},
                     indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
