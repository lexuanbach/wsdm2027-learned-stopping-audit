"""Faithful Stop-RAG comparison (Park, Cho & Lee, arXiv:2510.14337).

Implements the paper's method on our environment and logs:
  * MDP: state = question + retrieved docs so far; actions STOP / CONT;
    terminal-only rewards, r(s_t, STOP) = answer quality when answering at t.
  * Targets: full-width forward-view Q(lambda) —
      Qhat(s_t, STOP) = r(s_t, STOP)
      Qhat^{(n)}(s_t) = max({r(s_{t+k},STOP)}_{k=1..n-1}
                           u {Q_th(s_{t+n},STOP), Q_th(s_{t+n},CONT)})
      Qhat_lam(s_t, CONT) = (1-lam) sum_{n=1}^{T-t-1} lam^{n-1} Qhat^{(n)}
                             + lam^{T-t-1} Qhat^{(T-t)}
    with lambda cosine-decayed 1.0 -> 0.1 across fitted iterations
    (bootstrapped from the current network, as in the paper).
  * Model: DeBERTa-v3 encoder over "Q [SEP] d1 [SEP] ... [SEP] dt" with two
    FFN heads (STOP / CONT), hidden 4096 in the paper; we scale the head to
    the encoder width and disclose deviations.
  * Inference: stop at the first prefix where Q(STOP) - Q(CONT) > margin;
    margin swept on a validation split -> frontier; matched-accuracy point.

Faithful-adaptation notes (disclosed in the paper):
  - r(STOP) uses our single temperature-0 answer per prefix (their N=8
    stochastic samples average; at temp 0 the estimator is deterministic).
  - Encoder defaults to deberta-v3-base for laptop-scale training; pass
    --encoder microsoft/deberta-v3-large to match the paper exactly.

Phases:  --phase collect   (LLM trajectory collection, stores text states)
         --phase train     (Q(lambda) fitted training + evaluation, no LLM)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

import run_experiment as R

MEAS = R.OUT


def collect(args):
    import pandas as pd
    df = pd.read_parquet(args.parquet)
    df = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    rows = df.iloc[: args.n_train + args.n_test]
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        env = R.QuestionEnv(row)
        traj = R.run_react(env, args.model, max_steps=args.max_steps, k=3)
        out.append({
            "question": env.question,
            "gold": env.answer, "aliases": env.aliases, "type": env.type,
            "prefixes": [{
                "searches": p["searches"],
                "evidence": p["evidence"],
                "answer": p["answer"],
                "f1": env.f1(p["answer"]),
                "em": env.em(p["answer"]),
            } for p in traj],
        })
        if (i + 1) % 20 == 0:
            print(f"  collected {i+1}/{len(rows)}", flush=True)
            (MEAS / f"stoprag_traj_{args.tag}.json").write_text(json.dumps(out))
    (MEAS / f"stoprag_traj_{args.tag}.json").write_text(json.dumps(out))
    print(f"[collect] wrote {len(out)} trajectories")


# ---------------------------------------------------------------------------
def build_model(encoder_name):
    import torch
    from transformers import AutoModel, AutoTokenizer

    import torch
    tok = AutoTokenizer.from_pretrained(encoder_name)
    enc = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
    width = enc.config.hidden_size

    class QNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.stop_head = torch.nn.Sequential(
                torch.nn.Linear(width, 1024), torch.nn.GELU(),
                torch.nn.Linear(1024, 1))
            self.cont_head = torch.nn.Sequential(
                torch.nn.Linear(width, 1024), torch.nn.GELU(),
                torch.nn.Linear(1024, 1))

        def forward(self, **batch):
            h = self.enc(**batch).last_hidden_state[:, 0]
            return self.stop_head(h).squeeze(-1), self.cont_head(h).squeeze(-1)

    return tok, QNet()


def state_text(q, prefix):
    docs = " [SEP] ".join(prefix["evidence"]) if prefix["evidence"] else ""
    return f"{q} [SEP] {docs}" if docs else q


def train(args):
    import torch
    data = json.loads((MEAS / f"stoprag_traj_{args.tag}.json").read_text())
    n_tr = args.n_train
    train_d, test_d = data[:n_tr], data[n_tr:]
    # small validation split for the stopping margin
    val_d, train_d = train_d[-20:], train_d[:-20]
    dev = ("cpu" if getattr(args, "cpu", False)
           else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(getattr(args, "seed", 0))
    tok, net = build_model(args.encoder)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-5)

    def encode(texts):
        b = tok(texts, truncation=True, max_length=384, padding=True,
                return_tensors="pt")
        return {k: v.to(dev) for k, v in b.items()}

    def qvalues(d):
        """Current Q(STOP), Q(CONT) for every prefix of every trajectory."""
        net.eval()
        out = []
        with torch.no_grad():
            for c in d:
                texts = [state_text(c["question"], p) for p in c["prefixes"]]
                qs_l, qc_l = [], []
                for i in range(0, len(texts), 8):
                    s, co = net(**encode(texts[i:i + 8]))
                    qs_l += s.tolist(); qc_l += co.tolist()
                out.append((qs_l, qc_l))
        return out

    iters = args.fitted_iters
    for it in range(iters):
        lam = 0.1 + 0.9 * (math.cos(math.pi * it / max(iters - 1, 1)) + 1) / 2
        qv = qvalues(train_d)
        # build targets
        X, ys, yc = [], [], []
        for (qs_l, qc_l), c in zip(qv, train_d):
            r = [p["f1"] for p in c["prefixes"]]
            T = len(r)
            for t in range(T):
                X.append(state_text(c["question"], c["prefixes"][t]))
                ys.append(r[t])                       # Qhat(s_t, STOP)
                if t == T - 1:
                    yc.append(r[t])                   # horizon CONT = quality at T-1
                    continue
                nsteps = []
                for n in range(1, T - t):
                    cand = [r[t + k] for k in range(1, n)]
                    if t + n <= T - 1:
                        cand += [qs_l[t + n], qc_l[t + n]]
                    nsteps.append(max(cand) if cand else r[t + n])
                total, wsum = 0.0, 0.0
                for n, val in enumerate(nsteps[:-1], start=1):
                    w = (1 - lam) * lam ** (n - 1)
                    total += w * val; wsum += w
                wlast = lam ** (len(nsteps) - 1)
                total += wlast * nsteps[-1]; wsum += wlast
                yc.append(total / max(wsum, 1e-9))
        # one epoch of regression on both heads
        net.train()
        idx = np.random.default_rng(it).permutation(len(X))
        losses = []
        for i in range(0, len(idx), args.batch):
            sel = idx[i:i + args.batch].tolist()
            b = encode([X[j] for j in sel])
            ps, pc = net(**b)
            ts = torch.tensor([ys[j] for j in sel], device=dev, dtype=ps.dtype)
            tc = torch.tensor([yc[j] for j in sel], device=dev, dtype=pc.dtype)
            loss = torch.nn.functional.mse_loss(ps, ts) + \
                torch.nn.functional.mse_loss(pc, tc)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        print(f"[iter {it+1}/{iters}] lam={lam:.2f} loss={np.mean(losses):.4f}",
              flush=True)

    # ---- margin sweep on val, evaluate on test ----
    def frontier(d, margins):
        qv = qvalues(d)
        out = []
        for m in margins:
            ems, ss = [], []
            for (qs_l, qc_l), c in zip(qv, d):
                stop = len(c["prefixes"]) - 1
                for t in range(len(c["prefixes"]) - 1):
                    if qs_l[t] - qc_l[t] > m:
                        stop = t; break
                ems.append(c["prefixes"][stop]["em"])
                ss.append(c["prefixes"][stop]["searches"])
            out.append({"margin": m, "em": 100 * float(np.mean(ems)),
                        "s": float(np.mean(ss))})
        return out

    margins = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    ref_em = 100 * float(np.mean([c["prefixes"][-1]["em"] for c in test_d]))
    ref_s = float(np.mean([c["prefixes"][-1]["searches"] for c in test_d]))
    val_front = frontier(val_d, margins)
    vref = 100 * float(np.mean([c["prefixes"][-1]["em"] for c in val_d]))
    ok = [f for f in val_front if f["em"] >= vref - 1.0]
    m_star = (min(ok, key=lambda f: f["s"]) if ok
              else max(val_front, key=lambda f: f["em"]))["margin"]
    test_front = frontier(test_d, margins)
    picked = next(f for f in test_front if f["margin"] == m_star)
    # per-question stops at the validation-selected margin (exact calls =
    # 1 + 2 * stop under the shared answer-at-every-prefix loop)
    qv_test = qvalues(test_d)
    test_per_q = []
    for (qs_l, qc_l), c in zip(qv_test, test_d):
        stop = len(c["prefixes"]) - 1
        for t in range(len(c["prefixes"]) - 1):
            if qs_l[t] - qc_l[t] > m_star:
                stop = t; break
        test_per_q.append({"id": c.get("id"), "stop": int(stop),
                           "em": int(c["prefixes"][stop]["em"]),
                           "searches": int(c["prefixes"][stop]["searches"])})
    out = {"encoder": args.encoder, "fitted_iters": iters,
           "n_train_episodes": len(train_d), "n_val_episodes": len(val_d),
           "n_test": len(test_d), "test_per_q": test_per_q,
           "device": dev,
           "ref_em": round(ref_em, 1), "ref_s": round(ref_s, 2),
           "margin_star": m_star,
           "stoprag_em": round(picked["em"], 1),
           "stoprag_s": round(picked["s"], 2),
           "stoprag_save": round(100 * (1 - picked["s"] / ref_s)),
           "test_frontier": test_front}
    (MEAS / f"stoprag_result_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["collect", "train"], required=True)
    ap.add_argument("--parquet")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--n_train", type=int, default=90)
    ap.add_argument("--n_test", type=int, default=140)
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--fitted_iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="hotpot_qwen3b")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "collect":
        collect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
