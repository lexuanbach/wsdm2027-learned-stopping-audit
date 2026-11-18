"""Evaluate the published TASR predicate on VoI-Search trajectories.

TASR fires when (1) the normalized answer repeats and (2) a per-round
isotonic calibration of the first answer-token top-1/top-2 log-probability
margin exceeds its published fixed cutoff, 0.25.  Ollama returns logprobs on
the same answer call; replay here is needed only because older logs omitted
them.  Calibration uses training trajectories only and test outcomes never
select a parameter.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import re
import time
import urllib.request

import numpy as np
from sklearn.isotonic import IsotonicRegression

import run_experiment as rx


MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
URL = os.environ.get("VOISEARCH_OLLAMA_URL", "http://localhost:11434/api/chat")
TASR_THRESHOLD = 0.25


def norm(value):
    value = re.sub(r"[^a-z0-9 ]", " ", str(value).lower())
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_messages(question, evidence):
    ev = "\n".join(f"- {x}" for x in evidence) if evidence else "(no evidence)"
    return [
        {"role": "system", "content": (
            "You are a precise question-answering assistant. Give the shortest "
            "exact answer (a name, entity, number, or yes/no). Do not explain.")},
        {"role": "user", "content": (
            f"Evidence:\n{ev}\n\nQuestion: {question}\n"
            "Respond with exactly: Answer: <short answer>")},
    ]


def extract_answer(text):
    match = re.search(r"(?i)answer\s*[:\-]\s*(.+)", text)
    answer = match.group(1) if match else text
    return answer.strip().split("\n")[0].strip(" .\"'")


def first_answer_margin(content, logprobs):
    """Top-1 minus top-2 logprob at first non-space token after Answer:."""
    def item_margin(item):
        alternatives = item.get("top_logprobs", [])
        values = sorted({float(x["logprob"]) for x in alternatives},
                        reverse=True)
        if len(values) >= 2:
            return values[0] - values[1]
        selected = float(item.get("logprob", 0.0))
        other = [float(x["logprob"]) for x in alternatives
                 if x.get("token") != item.get("token")]
        return selected - max(other) if other else None

    prefix = ""
    answer_start = None
    for i, item in enumerate(logprobs):
        token = item.get("token", "")
        before = len(prefix)
        prefix += token
        if answer_start is None:
            match = re.search(r"(?i)answer\s*[:\-]\s*", prefix)
            if match:
                answer_start = match.end()
        if answer_start is not None and len(prefix) > answer_start:
            new_text = prefix[max(before, answer_start):]
            if re.search(r"\S", new_text):
                return item_margin(item)
    # The deployed prompt requests the literal marker, but this Qwen build
    # sometimes emits only the short answer.  In that case the first
    # non-whitespace generated token is exactly the answer-commitment token.
    for item in logprobs:
        if re.search(r"\S", item.get("token", "")):
            return item_margin(item)
    return None


def score_prefix(model, seed, temperature, question, prefix):
    payload = {
        "model": model,
        "messages": answer_messages(question, prefix["evidence"]),
        "stream": False,
        "logprobs": True,
        "top_logprobs": 2,
        "options": {"temperature": temperature, "num_predict": 40,
                    "num_ctx": 4096, "seed": seed},
    }
    started = time.perf_counter()
    request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read())
    content = result["message"]["content"].strip()
    return {
        "answer": extract_answer(content),
        "margin": first_answer_margin(content, result.get("logprobs", [])),
        "input_tokens": result.get("prompt_eval_count", 0),
        "output_tokens": result.get("eval_count", 0),
        "latency_seconds": time.perf_counter() - started,
    }


def rows_from_run(run, cache):
    train = cache.get("traces", [])
    test = run["per_q"]
    return [("train", row) for row in train] + [("test", row) for row in test]


def original_answer(prefix):
    return prefix.get("answer", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="corpus_guarded_qwen3b")
    args = parser.parse_args()
    run = json.loads((MEAS / f"run_{args.tag}.json").read_text())
    cache = pickle.loads((MEAS / f"critic_data_{args.tag}.pkl").read_bytes())
    model = run["config"]["model"]
    seed = int(run["config"].get("llm_seed", 7))
    temperature = float(run["config"].get("llm_temperature", 0.0))
    output = MEAS / f"tasr_{args.tag}.json"
    partial = MEAS / f"tasr_{args.tag}.partial.json"
    scored = json.loads(partial.read_text()).get("rows", []) if partial.exists() else []
    done = {(row["split"], row["id"]) for row in scored}
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
              "latency_seconds": 0.0}
    for split, row in rows_from_run(run, cache):
        key = (split, str(row["id"]))
        if key in done:
            continue
        prefix_scores = []
        for prefix in row["trajectory"]:
            measured = score_prefix(model, seed, temperature,
                                    row["question"], prefix)
            measured["matches_logged_answer"] = (
                norm(measured["answer"]) == norm(original_answer(prefix)))
            prefix_scores.append(measured)
            totals["calls"] += 1
            totals["input_tokens"] += measured["input_tokens"]
            totals["output_tokens"] += measured["output_tokens"]
            totals["latency_seconds"] += measured["latency_seconds"]
        scored.append({"split": split, "id": str(row["id"]),
                       "prefix": prefix_scores})
        if len(scored) % 20 == 0:
            partial.write_text(json.dumps({"tag": args.tag, "rows": scored}, indent=1))
            print(f"[{len(scored)}] rescored", flush=True)

    scored_map = {(row["split"], row["id"]): row for row in scored}
    train_traces = cache["traces"]
    split_rng = np.random.default_rng(41)
    split_order = np.arange(len(train_traces))
    split_rng.shuffle(split_order)
    n_certificate = max(1, int(round(
        run["config"].get("certificate_calibration_fraction", 0.25)
        * len(train_traces))))
    isotonic_traces = [train_traces[i] for i in split_order[:-n_certificate]]
    calibrators = []
    max_rounds = max(len(row["trajectory"])
                     for _, row in rows_from_run(run, cache))
    for t in range(max_rounds):
        x, y = [], []
        for row in isotonic_traces:
            score = scored_map[("train", str(row["id"]))]["prefix"][t]
            if score["margin"] is not None:
                x.append(score["margin"])
                y.append(max(rx.em(score["answer"], gold)
                             for gold in row["gold_answers"]))
        if len(set(x)) >= 2 and len(set(y)) >= 2:
            iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
            calibrators.append(iso)
        else:
            calibrators.append(float(np.mean(y)) if y else 0.0)

    per_q = []
    for row in run["per_q"]:
        values = scored_map[("test", str(row["id"]))]["prefix"]
        calibrated = []
        for t, score in enumerate(values):
            margin = score["margin"]
            model_or_constant = calibrators[t]
            if margin is None:
                calibrated.append(None)
            elif hasattr(model_or_constant, "predict"):
                calibrated.append(float(model_or_constant.predict([margin])[0]))
            else:
                calibrated.append(float(model_or_constant))
        stop = len(values) - 1
        for t in range(1, len(values)):
            if (norm(values[t]["answer"])
                    == norm(values[t - 1]["answer"])
                    and calibrated[t] is not None
                    and calibrated[t] > TASR_THRESHOLD):
                stop = t
                break
        replay_answer = values[stop]["answer"]
        gold_answers = row.get("gold_answers", [])
        replay_em = max(rx.em(replay_answer, gold) for gold in gold_answers)
        replay_f1 = max(rx.f1(replay_answer, gold) for gold in gold_answers)
        deployment_cost = {
            "calls": 1 + 2 * stop,
            "input_tokens": sum(x["input_tokens"] for x in values[:stop + 1]),
            "output_tokens": sum(x["output_tokens"] for x in values[:stop + 1]),
            "llm_latency_seconds": sum(x["latency_seconds"]
                                       for x in values[:stop + 1]),
            "retrieval_latency_seconds": sum(
                float(prefix.get("retrieval_latency_seconds", 0.0))
                for prefix in row["trajectory"][:stop + 1]),
        }
        for prefix in row["trajectory"][1:stop + 1]:
            query_usage = prefix.get("query_usage", {})
            deployment_cost["input_tokens"] += float(query_usage.get("in", 0))
            deployment_cost["output_tokens"] += float(query_usage.get("out", 0))
            deployment_cost["llm_latency_seconds"] += float(
                query_usage.get("latency_seconds", 0.0))
        per_q.append({"id": str(row["id"]), "stop": stop,
                      "answer": replay_answer, "em": replay_em,
                      "f1": replay_f1, "deployment_cost": deployment_cost,
                      "calibrated_margin": calibrated,
                      "raw_margin": [x["margin"] for x in values]})

    totals = {
        "calls": sum(len(row["prefix"]) for row in scored),
        "input_tokens": sum(prefix["input_tokens"] for row in scored
                            for prefix in row["prefix"]),
        "output_tokens": sum(prefix["output_tokens"] for row in scored
                             for prefix in row["prefix"]),
        "latency_seconds": sum(prefix["latency_seconds"] for row in scored
                               for prefix in row["prefix"]),
    }
    match_flags = [prefix["matches_logged_answer"] for row in scored
                   for prefix in row["prefix"]]
    test_match_flags = [prefix["matches_logged_answer"] for row in scored
                        if row["split"] == "test" for prefix in row["prefix"]]
    result = {
        "method": "TASR_AS_m25",
        "rule": "normalized consecutive answer repeat AND per-round isotonic calibrated first-answer-token margin > 0.25",
        "threshold": TASR_THRESHOLD,
        "selection": "published fixed threshold; isotonic calibrators fit on training episodes only",
        "isotonic_calibration_episodes": len(isotonic_traces),
        "excluded_certificate_episodes": n_certificate,
        "split_seed": 41,
        "tag": args.tag,
        "model": model,
        "llm_seed": seed,
        "llm_temperature": temperature,
        "replay_answer_match_rate": float(np.mean(match_flags)),
        "test_replay_answer_match_rate": float(np.mean(test_match_flags)),
        "replay_cost": totals,
        "deployment_cost_note": (
            "Logprobs accompany the existing answer call. TASR quality and "
            "answer-call token/latency costs use the logprob-enabled replay; "
            "queries and evidence remain fixed. The replay calls are "
            "measurement overhead, not additional deployment calls."),
        "per_q": per_q,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    if partial.exists():
        partial.unlink()
    print(json.dumps({"n": len(per_q),
                      "em": 100 * float(np.mean([x["em"] for x in per_q])),
                      "searches": float(np.mean([x["stop"] for x in per_q])),
                      "answer_match": result["replay_answer_match_rate"]}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
