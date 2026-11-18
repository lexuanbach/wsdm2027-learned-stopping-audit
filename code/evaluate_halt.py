"""Matched generated-claim HALT condition on the frozen trajectories.

HALT stops when the cumulative evidence covers every claim the question
requires (Roh et al., 2026).  The released system pairs an LLM claim generator
with a *fine-tuned* verifier adapter that needs a 24GB CUDA GPU, which this
host does not have, so we run the closest matched condition we can:

  * claims are **generated**, not gold: one extra call to the run's own frozen
    backbone per question, prompted for the atomic facts the question needs
    (at most ``--max-claims``);
  * coverage is verified by a fixed public NLI cross-encoder
    (cross-encoder/nli-MiniLM2-L6-H768, pinned revision) between the
    cumulative evidence and each claim -- the same verifier family and the
    same checkpoint the semantic-feature probe uses;
  * the rule stops at the first prefix where *every* claim is covered
    (min entailment >= tau), otherwise at the budget;
  * tau is selected on the run's own policy-training chains under the paper's
    frozen band rule (cheapest grid point within 1 EM of the training
    full-budget reference), never on test outcomes;
  * costs are the paper's ledger: exact LLM calls 1 + 2*stop for the replayed
    trajectory, **plus one claim-generation call**, which we report separately
    and inside the total.  Verifier inferences are local and are reported as a
    separate count, exactly as HALT's verifier is not an LLM call.

This is therefore a *matched generated-claim HALT condition*, not a
reproduction of HALT's trained verifier; the manuscript says so.  Nothing here
is selected on test outcomes.

    python3 evaluate_halt.py --tag corpus_guarded_qwen3b
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import re
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import run_experiment as rx

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
VERIFIER = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
# Generated hop claims come out both as declaratives ("The year X was
# published.") and as retrieval questions ("Which album ...?").  Entailment is
# ill-defined for the latter, so we run the coverage test with two fixed public
# verifiers and report both: the NLI cross-encoder (HALT's entailment family)
# and a relevance cross-encoder (support/coverage family, the same checkpoint
# the Wikipedia hybrid reranks with).  tau is selected separately for each on
# the training chains; the conclusion has to hold for both.
RELEVANCE = "cross-encoder/ms-marco-MiniLM-L6-v2"
TAU_GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
BAND_EM = 1.0
CLAIM_PROMPT = (
    "You decompose a multi-hop question into the facts that must be RETRIEVED "
    "before it can be answered. Never answer the question. Each line states "
    "what must be found and mentions an entity from the question. Output at "
    "most {k} lines, no numbering, no explanation.\n\n"
    "Question: Which magazine was started first, Arthur's Magazine or First "
    "for Women?\nFacts:\nThe year Arthur's Magazine was first published.\n"
    "The year First for Women was first published.\n\n"
    "Question: The Oberoi family is part of a hotel company that has a head "
    "office in what city?\nFacts:\nThe hotel company the Oberoi family is "
    "part of.\nThe city where that hotel company has its head office.\n\n"
    "Question: {q}\nFacts:")
JUNK = ("unknown part", "blank", "no explanation", "n/a")


def generate_claims(question, model, max_claims):
    text = rx.llm([{"role": "user",
                    "content": CLAIM_PROMPT.format(k=max_claims, q=question)}],
                  model, max_tokens=160)
    claims = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        low = line.lower()
        if (len(line) >= 8 and not low.startswith(("fact", "answer:", "question:"))
                and not any(bad in low for bad in JUNK)):
            claims.append(line)
    if not claims:                       # conservative fallback: the question
        claims = [question.strip()]
    return claims[:max_claims], text


def score_pairs(pairs, checkpoint, revision, batch_size, device_name):
    """Coverage score for (premise, claim) pairs under a fixed cross-encoder.

    NLI checkpoints score the entailment class; the relevance cross-encoder
    emits a single logit, which we squash with a sigmoid.  Both are read-only,
    pinned, and never fine-tuned here.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, revision=revision, local_files_only=True)
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device).eval()
    n_labels = model.config.num_labels
    label = None
    if n_labels > 1:
        label = next(i for i, name in model.config.id2label.items()
                     if str(name).lower() == "entailment")
    out = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            batch = tokenizer([p for p, _ in chunk], [h for _, h in chunk],
                              padding=True, truncation=True, max_length=384,
                              return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            if label is None:
                values = torch.sigmoid(logits[:, 0])
            else:
                values = torch.softmax(logits, dim=-1)[:, label]
            out.extend(values.detach().cpu().numpy().tolist())
    return out


def coverage_curve(trajectory, claims, scores_by_prefix):
    """min-over-claims entailment at each prefix (HALT: all claims covered)."""
    return [float(min(scores_by_prefix[t])) if claims else 1.0
            for t in range(len(trajectory))]


def halt_stop(curve, tau):
    for t, value in enumerate(curve[:-1]):
        if value >= tau:
            return t
    return len(curve) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corpus_guarded_qwen3b")
    ap.add_argument("--max-claims", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = ap.parse_args()

    run = json.loads((MEAS / f"run_{args.tag}.json").read_text())
    cache = pickle.loads((MEAS / f"critic_data_{args.tag}.pkl").read_bytes())
    model = run["config"]["model"]
    rx._LLM_SEED = run["config"].get("llm_seed", 7)
    rx._LLM_TEMPERATURE = 0.0                     # claim generation is greedy
    max_steps = run["config"]["max_steps"]
    t0 = time.time()

    # ---- 1. claims (one extra LLM call per question, train + test) -------- #
    claim_path = MEAS / f"halt_claims_{args.tag}.json"
    if claim_path.exists():
        claims_by = {(r["split"], r["id"]): r["claims"]
                     for r in json.loads(claim_path.read_text())["rows"]}
        print(f"[claims] reusing {len(claims_by)} cached claim sets", flush=True)
    else:
        claims_by, rows = {}, []
        units = ([("train", r) for r in cache["traces"]]
                 + [("test", r) for r in run["per_q"]])
        for i, (split, row) in enumerate(units):
            claims, raw = generate_claims(row["question"], model, args.max_claims)
            claims_by[(split, str(row["id"]))] = claims
            rows.append({"split": split, "id": str(row["id"]),
                         "claims": claims, "raw": raw})
            if (i + 1) % 50 == 0:
                print(f"  [claims {i+1}/{len(units)}] {time.time()-t0:.0f}s",
                      flush=True)
        claim_path.write_text(json.dumps(
            {"model": model, "prompt": CLAIM_PROMPT, "max_claims": args.max_claims,
             "llm_seed": rx._LLM_SEED, "temperature": 0.0,
             "n_claim_generation_calls": len(rows), "rows": rows}, indent=1) + "\n")
        print(f"[claims] {len(rows)} generated in {time.time()-t0:.0f}s", flush=True)

    # ---- 2. coverage curves under both fixed verifiers ------------------ #
    units = ([("train", r) for r in cache["traces"]]
             + [("test", r) for r in run["per_q"]])
    score_path = MEAS / f"halt_scores_{args.tag}.json"
    if score_path.exists():
        cached = json.loads(score_path.read_text())
        curves = {v: {(r["split"], r["id"]): r[v] for r in cached["rows"]}
                  for v in ["nli", "relevance"]}
        n_pairs = cached["n_verifier_pairs"]
        print(f"[verifier] reusing cached curves for {len(cached['rows'])} units",
              flush=True)
    else:
        pairs, index = [], []
        for split, row in units:
            claims = claims_by[(split, str(row["id"]))]
            for t, prefix in enumerate(row["trajectory"]):
                premise = " ".join(prefix["evidence"]) or "No retrieved evidence."
                for claim in claims:
                    index.append((split, str(row["id"]), t))
                    pairs.append((premise, claim))
        n_pairs = len(pairs)
        print(f"[verifier] {n_pairs} claim-prefix pairs x 2 checkpoints", flush=True)
        curves = {}
        for name, ckpt, rev in [("nli", VERIFIER, REVISION),
                                ("relevance", RELEVANCE, None)]:
            values = score_pairs(pairs, ckpt, rev, args.batch_size, args.device)
            grouped = {}
            for (split, qid, t), value in zip(index, values):
                grouped.setdefault((split, qid), {}).setdefault(t, []).append(value)
            curves[name] = {
                (split, str(row["id"])): [
                    float(min(grouped[(split, str(row["id"]))][t]))
                    for t in range(len(row["trajectory"]))]
                for split, row in units}
            print(f"  [{name}] scored {time.time()-t0:.0f}s", flush=True)
        score_path.write_text(json.dumps(
            {"verifiers": {"nli": {"checkpoint": VERIFIER, "revision": REVISION,
                                   "score": "entailment probability"},
                           "relevance": {"checkpoint": RELEVANCE,
                                         "score": "sigmoid of the relevance logit"}},
             "rule": "coverage(t) = min over generated claims of score(evidence_t, claim)",
             "n_verifier_pairs": n_pairs,
             "rows": [{"split": k[0], "id": k[1],
                       "nli": curves["nli"][k], "relevance": curves["relevance"][k]}
                      for k in curves["nli"]]}, indent=1) + "\n")

    # ---- 3. select tau on the policy-training chains only ---------------- #
    X = np.asarray(cache["X_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    all_chains = rx.chain_indices(X)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    fit_ids = order[:-ncal]
    train_chains = [all_chains[i] for i in fit_ids]
    train_ids = [str(cache["traces"][i]["id"]) for i in fit_ids]
    reference = 100 * float(np.mean([1 - y_err[ch[-1]] for ch in train_chains]))

    def select_tau(name):
        train_curves = [curves[name][("train", qid)] for qid in train_ids]
        frontier = []
        for tau in TAU_GRID:
            stops = [halt_stop(curve, tau) for curve in train_curves]
            em = 100 * float(np.mean([1 - y_err[ch[t]]
                                      for ch, t in zip(train_chains, stops)]))
            frontier.append({"tau": tau, "em": round(em, 3),
                             "searches": round(float(np.mean(stops)), 3)})
        eligible = [f for f in frontier if f["em"] >= reference - BAND_EM]
        if eligible:
            picked = min(eligible, key=lambda f: (f["searches"], -f["em"]))
            reason = "cheapest_within_1em_training_band"
        else:
            best = max(f["em"] for f in frontier)
            picked = min([f for f in frontier if f["em"] == best],
                         key=lambda f: f["searches"])
            reason = "fallback_max_training_em"
        return picked["tau"], reason, frontier

    # ---- 4. replay on the untouched test questions ----------------------- #
    def replay(name, tau):
        rows = []
        for row in run["per_q"]:
            curve = curves[name][("test", str(row["id"]))]
            stop = halt_stop(curve, tau)
            usage = {"calls": 1 + 2 * stop, "in": 0.0, "out": 0.0,
                     "latency_seconds": 0.0, "retrieval_latency_seconds": 0.0}
            for prefix in row["trajectory"][:stop + 1]:
                for part in ["query_usage", "answer_usage"]:
                    for key, value in prefix.get(part, {}).items():
                        if key == "in":
                            usage["in"] += float(value)
                        elif key == "out":
                            usage["out"] += float(value)
                        elif key == "latency_seconds":
                            usage["latency_seconds"] += float(value)
                usage["retrieval_latency_seconds"] += float(
                    prefix.get("retrieval_latency_seconds", 0.0))
            rows.append({
                "id": str(row["id"]), "stop": int(stop),
                "em": int(row["prefix_em"][stop]),
                "f1": float(row["prefix_f1"][stop]),
                "coverage": curve,
                "n_claims": len(claims_by[("test", str(row["id"]))]),
                "deployment_cost": {
                    "calls_without_claim_generation": usage["calls"],
                    "calls": usage["calls"] + 1,
                    "input_tokens": usage["in"], "output_tokens": usage["out"],
                    "llm_latency_seconds": usage["latency_seconds"],
                    "retrieval_latency_seconds": usage["retrieval_latency_seconds"]},
            })
        return rows

    variants = {}
    for name in ["nli", "relevance"]:
        tau, reason, frontier = select_tau(name)
        rows = replay(name, tau)
        variants[name] = {
            "verifier": VERIFIER if name == "nli" else RELEVANCE,
            "tau": tau, "selection_reason": reason, "training_frontier": frontier,
            "em": round(100 * float(np.mean([r["em"] for r in rows])), 2),
            "f1": round(100 * float(np.mean([r["f1"] for r in rows])), 2),
            "mean_stop": round(float(np.mean([r["stop"] for r in rows])), 3),
            "mean_calls": round(float(np.mean(
                [r["deployment_cost"]["calls"] for r in rows])), 3),
            "mean_calls_without_claim_generation": round(float(np.mean(
                [r["deployment_cost"]["calls_without_claim_generation"]
                 for r in rows])), 3),
            "mean_input_tokens": round(float(np.mean(
                [r["deployment_cost"]["input_tokens"] for r in rows])), 1),
            "mean_wall_seconds": round(float(np.mean(
                [r["deployment_cost"]["llm_latency_seconds"]
                 + r["deployment_cost"]["retrieval_latency_seconds"]
                 for r in rows])), 3),
            "per_q": rows,
        }
        print(f"[{name}] tau={tau} ({reason}) -> EM {variants[name]['em']} "
              f"at {variants[name]['mean_calls']} calls", flush=True)

    result = {
        "method": "HALT_generated_claim_matched_condition",
        "not_a_reproduction": (
            "Claims are generated by the run's own frozen backbone and coverage is "
            "judged by fixed public cross-encoders; HALT's released fine-tuned "
            "verifier adapter needs a 24GB CUDA GPU unavailable here. Retrieval "
            "stack, trajectories, backbone, and cost ledger are matched."),
        "tag": args.tag, "model": model, "claim_prompt": CLAIM_PROMPT,
        "max_claims": args.max_claims, "tau_grid": TAU_GRID,
        "training_reference_em": round(reference, 3),
        "n_policy_training_episodes": len(train_chains),
        "n_test": len(run["per_q"]), "n_verifier_pairs": n_pairs,
        "mean_claims_per_question": round(float(np.mean(
            [len(v) for v in claims_by.values()])), 3),
        "claim_generation_calls_per_question": 1,
        "cost_note": ("calls = 1 + 2*stop for the replayed trajectory plus one "
                      "claim-generation call; verifier inferences are local and "
                      "counted separately as n_verifier_pairs"),
        "primary_verifier": "nli",
        "variants": variants,
        "wall_seconds": round(time.time() - t0, 1),
    }
    dest = MEAS / f"halt_{args.tag}.json"
    dest.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"tag": result["tag"],
                      "mean_claims": result["mean_claims_per_question"],
                      "variants": {k: {q: v[q] for q in
                                       ["tau", "em", "mean_calls", "mean_stop"]}
                                   for k, v in variants.items()}}, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
