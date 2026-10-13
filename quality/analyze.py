"""Phân tích kết quả đo chất lượng (P06 và mọi bước tối ưu sau). Chạy ở LOCAL, không cần GPU.

    .venv/bin/python quality/analyze.py --out results/gd1/06-chat-luong-baseline/<run>        → summary.md, summary.json
    .venv/bin/python quality/analyze.py --compare <run_baseline> <run_variant>                 → độ gần + flips (sau này)

Chỉ số và căn cứ (docs/references nhóm E):
- Điểm tác vụ + CI 95% = p ± 1.96·sqrt(p(1−p)/n) (Miller 2024, E3, công thức 3).
- So hai cấu hình trên cùng câu hỏi: hiệu theo từng câu (paired SE) + flips đúng↔sai (Miller E3 mục 4.2; Dutta 2024 E4).
- Độ gần token (E5 llama.cpp, E4 Dutta, E7 Thinking Machines):
  - sinh tự do: % output giống hệt, vị trí bắt đầu lệch; trên đoạn chung (cùng ngữ cảnh) Δp của token đã chọn và KL xấp xỉ;
  - teacher forcing: same top-1 %, Δp của token tham chiếu, KL xấp xỉ.
  KL tính từ top-20 nên là XẤP XỈ (E6): phần ngoài top-20 gộp thành một "đuôi"; báo kèm độ phủ = tổng xác suất top-20.
- FinQA: dung sai tương đối 1% như FinLFQA (E12), chấp nhận dạng phần trăm (0.1446 ↔ 14.46); báo thêm 0.1% và 5%.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import common as C  # noqa: E402

TOLS = (0.001, 0.01, 0.05)
MAIN_TOL = 0.01


# ---------- đọc kết quả ----------

def unit_recs(run, uid):
    d = Path(run) / "units" / uid
    if not (d / "done.json").exists():
        return None
    recs = {}
    for f in sorted((d / "chunks").glob("*.json.gz")):
        for r in C.read_json(f):
            recs[r["idx"]] = r
    return recs


def ci95(p, n):
    se = math.sqrt(max(p * (1 - p), 0) / n) if n else math.nan
    return se, (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


# ---------- độ gần token ----------

def kl_topk(top_p, top_q):
    """KL(P||Q) xấp xỉ từ top-k của P và Q (list [id, logprob]). Token của P không có trong top-k của Q được gán
    xác suất = min(xác suất nhỏ nhất trong top-k của Q, phần đuôi của Q); phần đuôi của P so với phần đuôi của Q.
    Trả về (kl, độ phủ của P)."""
    P = {int(t): math.exp(l) for t, l in top_p}
    Q = {int(t): math.exp(l) for t, l in top_q}
    sp, sq = min(sum(P.values()), 1.0), min(sum(Q.values()), 1.0)
    tail_q = max(1.0 - sq, 1e-12)
    floor_q = min(min(Q.values()) if Q else 1e-12, tail_q)
    kl = sum(p * math.log(p / max(Q.get(t, floor_q), 1e-12)) for t, p in P.items() if p > 0)
    tail_p = max(1.0 - sp, 0.0)
    if tail_p > 1e-9:
        kl += tail_p * math.log(tail_p / tail_q)
    return max(kl, 0.0), sp


def pct(xs, qs):
    xs = np.asarray(xs, float)
    return {f"p{q:g}": float(np.percentile(xs, q)) for q in qs} if len(xs) else {}


def gen_closeness(A, B):
    """A, B: {idx: record gen}. So từng prompt chung."""
    exact, div_pos, lens, dp, kl, cov, margins = 0, [], [], [], [], [], []
    n = 0
    for idx in sorted(set(A) & set(B)):
        a, b = A[idx], B[idx]
        ai, bi = a.get("out_ids") or [], b.get("out_ids") or []
        n += 1
        L = min(len(ai), len(bi))
        j = next((k for k in range(L) if ai[k] != bi[k]), None)
        if j is None and len(ai) == len(bi):
            exact += 1
            j_end = L
        else:
            j = L if j is None else j
            div_pos.append(j)
            lens.append(max(len(ai), len(bi)))
            j_end = j
            ta = a.get("top") or []
            if j < len(ta) and len(ta[j]) >= 2:
                margins.append(math.exp(ta[j][0][1]) - math.exp(ta[j][1][1]))
        la, lb, ta, tb = a.get("lp") or [], b.get("lp") or [], a.get("top") or [], b.get("top") or []
        for k in range(min(j_end, len(la), len(lb))):
            if la[k] is not None and lb[k] is not None:
                dp.append(math.exp(lb[k]) - math.exp(la[k]))
            if k < len(ta) and k < len(tb) and ta[k] and tb[k]:
                v, c = kl_topk(ta[k], tb[k])
                kl.append(v)
                cov.append(c)
    divs = np.asarray(div_pos, float)
    return {"n": n, "exact_match": exact / n if n else math.nan, "n_diverged": len(div_pos),
            "divergence_pos": pct(divs, (10, 50, 90)) if len(divs) else {},
            "survive_to": {t: float(np.mean([1.0] * exact + [float(x >= t) for x in div_pos])) if n else math.nan
                           for t in (16, 64, 128, 256)},
            "margin_at_divergence": pct(margins, (50, 90)) if margins else {},
            "common_prefix_tokens": len(dp),
            "dp": summarize_dp(dp), "kl": {"mean": float(np.mean(kl)) if kl else math.nan, **pct(kl, (50, 99, 99.9))},
            "coverage_top20": {"mean": float(np.mean(cov)) if cov else math.nan, **pct(cov, (1, 50))}}


def summarize_dp(dp):
    if not dp:
        return {}
    x = np.asarray(dp, float)
    return {"mean": float(x.mean()), "rms": float(np.sqrt((x ** 2).mean())), **pct(x, (1, 50, 99)),
            "max_abs": float(np.abs(x).max())}


def tf_closeness(A, B):
    """Teacher forcing: cùng ngữ cảnh ở mọi vị trí → same top-1, Δp của token tham chiếu, KL xấp xỉ."""
    same, tot, dp, kl, cov = 0, 0, [], [], []
    for idx in sorted(set(A) & set(B)):
        a, b = A[idx], B[idx]
        for k in range(min(len(a["lp"]), len(b["lp"]))):
            ta, tb = a["top"][k], b["top"][k]
            if ta and tb:
                tot += 1
                same += ta[0][0] == tb[0][0]
                v, c = kl_topk(ta, tb)
                kl.append(v)
                cov.append(c)
            if a["lp"][k] is not None and b["lp"][k] is not None:
                dp.append(math.exp(b["lp"][k]) - math.exp(a["lp"][k]))
    p = same / tot if tot else math.nan
    se, _ = ci95(p, tot)
    return {"positions": tot, "same_top1": p, "same_top1_se": se, "dp": summarize_dp(dp),
            "kl": {"mean": float(np.mean(kl)) if kl else math.nan, **pct(kl, (50, 99, 99.9))},
            "coverage_top20": {"mean": float(np.mean(cov)) if cov else math.nan, **pct(cov, (1, 50))}}


# ---------- FinQA ----------

NUM = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?|-?\$?\.\d+%?")


def parse_answer(text):
    """Giá trị trên dòng 'Answer:' cuối cùng; nếu không có thì lấy số cuối của câu trả lời."""
    text = re.sub(r"\[Documents?\s*[\d,\s]+\]", " ", text)  # bỏ trích dẫn [Document 2] (prompt W2 yêu cầu trích dẫn)
    m = list(re.finditer(r"Answer:\s*(.+)", text))
    seg = m[-1].group(1).strip() if m else text[-200:]
    low = seg.lower()
    if re.match(r"^\W*(yes|no)\b", low):
        return re.match(r"^\W*(yes|no)", low).group(1)
    nums = NUM.findall(seg)
    if not nums:
        return None
    s = nums[0] if m else nums[-1]
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    v = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "").replace("%", "").lstrip("-")
    try:
        x = float(v)
    except ValueError:
        return None
    return -x if neg else x


def finqa_correct(pred, gold, tol):
    if isinstance(gold, str):
        return isinstance(pred, str) and pred == gold.strip().lower()
    if not isinstance(pred, float):
        return False
    for g in (gold, gold * 100):  # đáp án gốc ghi phần trăm dạng phân số (0.1446 = 14.46%)
        if abs(pred - g) <= tol * abs(g) or (g == 0 and abs(pred) <= 1e-6):
            return True
    return False


def score_finqa(run, uid="finqa-c32-r1"):
    recs = unit_recs(run, uid)
    if recs is None:
        return None
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(ROOT / "data/quality/finqa_rag.jsonl")}
    per = {}
    for r in recs.values():
        pred = parse_answer(r["text"])
        g = gold[r["id"]]["gold"]
        per[r["id"]] = {"pred": pred, "gold": g, **{f"ok@{t:g}": finqa_correct(pred, g, t) for t in TOLS},
                        "finish": r.get("finish_reason"), "has_answer_line": "Answer:" in r["text"]}
    n = len(per)
    out = {"n": n, "no_answer_line": sum(not v["has_answer_line"] for v in per.values()),
           "unparsed": sum(v["pred"] is None for v in per.values()),
           "truncated": sum(v["finish"] == "length" for v in per.values())}
    for t in TOLS:
        p = sum(v[f"ok@{t:g}"] for v in per.values()) / n
        se, ci = ci95(p, n)
        out[f"acc@{t:g}"] = {"acc": p, "se": se, "ci95": ci}
    return out, per


# ---------- lm-eval ----------

# chỉ số đầu tiên của mỗi task = chỉ số chính (chốt trước khi đo, plan P06 D2): gsm8k dùng flexible-extract (số cuối),
# ít phụ thuộc định dạng "#### N" của few-shot; strict-match báo kèm.
LM_METRIC = {"gsm8k": [("exact_match,flexible-extract", "exact_match"), ("exact_match,strict-match", "exact_match")],
             "ifeval": [("prompt_level_strict_acc,none", "prompt_level_strict_acc"),
                        ("inst_level_strict_acc,none", None), ("prompt_level_loose_acc,none", "prompt_level_loose_acc")],
             "mmlu_pro": [("exact_match,custom-extract", "exact_match")]}


def lm_results(run, uid, task):
    d = Path(run) / "units" / uid
    if not (d / "done.json").exists():
        return None
    res = json.loads(sorted((d / "out").rglob("results_*.json"))[-1].read_text())
    # samples_*.jsonl có MỘT DÒNG CHO MỖI BỘ LỌC của mỗi câu (gsm8k: strict-match và flexible-extract) → tách theo bộ lọc
    per = {}
    for f in sorted((d / "out").rglob("samples_*.jsonl")):
        sub = re.sub(r"^samples_(.*)_\d{4}-\d{2}-\d{2}T.*$", r"\1", f.stem)
        for line in open(f):
            s = json.loads(line)
            per.setdefault(s.get("filter", "none"), {})[f"{sub}:{s['doc_id']}"] = s
    out = {"n": max((len(v) for v in per.values()), default=0), "task_results": {}, "per_doc": {}}
    for mkey, skey in LM_METRIC[task]:
        if task in res["results"] and mkey in res["results"][task]:
            out["task_results"][mkey] = res["results"][task][mkey]
        flt = mkey.split(",")[1]
        rows = per.get(flt, {})
        if skey:
            vals = [float(s[skey]) for s in rows.values() if skey in s and isinstance(s[skey], (int, float, bool))]
            if vals:
                p = float(np.mean(vals))
                se, ci = ci95(p, len(vals))
                out[f"{skey}@{flt}"] = {"acc": p, "se": se, "ci95": ci, "n": len(vals)}
                out["per_doc"][mkey] = {k: float(s[skey]) for k, s in rows.items() if skey in s}
    if task == "mmlu_pro":
        subs = {}
        for k, s in per.get("custom-extract", {}).items():
            subs.setdefault(k.split(":")[0], []).append(float(s.get("exact_match", 0)))
        out["by_subject"] = {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(subs.items())}
    out["lm_eval_version"] = res.get("lm_eval_version")
    out["config_model_args"] = res.get("config", {}).get("model_args")
    return out


# ---------- so hai cấu hình (dùng sau P06) ----------

def paired(a, b):
    """a, b: {câu: 0/1}. Hiệu B − A, paired SE (Miller), flips đúng→sai / sai→đúng (Dutta)."""
    keys = sorted(set(a) & set(b))
    x = np.array([b[k] - a[k] for k in keys], float)
    n = len(keys)
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {"n": n, "diff": float(x.mean()), "se_paired": se, "ci95": (float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)),
            "flips_correct_to_wrong": int(sum(1 for k in keys if a[k] > b[k])),
            "flips_wrong_to_correct": int(sum(1 for k in keys if a[k] < b[k])),
            "flips_pct": float(np.mean([a[k] != b[k] for k in keys])) if n else math.nan}


# ---------- báo cáo ----------

def f(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def report(run):
    run = Path(run)
    js, md = {"run": str(run)}, [f"# Tổng hợp chất lượng `{run}`\n"]
    refs = {u: unit_recs(run, u) for u in ("ref-gen-c8-r1", "ref-gen-c8-r2", "ref-gen-c64-r1")}
    tfs = {u: unit_recs(run, u) for u in ("ref-tf-c1-r1", "ref-tf-c8-r1")}
    src = {json.loads(x)["idx"]: json.loads(x)["source"] for x in open(ROOT / "data/quality/reference.jsonl")}

    md.append("## Mức tự lệch của vLLM mặc định (sinh tự do, greedy, 256 token)\n")
    md.append("| So sánh | prompt | giống hệt | lệch: vị trí p10 / p50 / p90 | còn giống tới token 64 / 256 | "
              "Δp đoạn chung: mean / p1 / p99 | KL≈ mean / p99 | độ phủ top-20 | chênh top1−top2 ở điểm lệch p50 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    js["gen"] = {}
    for name, a, b in (("c8 r1 vs c8 r2 (cùng mức tải)", "ref-gen-c8-r1", "ref-gen-c8-r2"),
                       ("c8 r1 vs c64 r1 (khác mức tải)", "ref-gen-c8-r1", "ref-gen-c64-r1")):
        if refs[a] is None or refs[b] is None:
            continue
        g = gen_closeness(refs[a], refs[b])
        js["gen"][name] = g
        dvp = g["divergence_pos"]
        md.append(f"| {name} | {g['n']} | {g['exact_match'] * 100:.1f}% | {dvp.get('p10', math.nan):.0f} / "
                  f"{dvp.get('p50', math.nan):.0f} / {dvp.get('p90', math.nan):.0f} | "
                  f"{g['survive_to'][64] * 100:.1f}% / {g['survive_to'][256] * 100:.1f}% | "
                  f"{f(g['dp'].get('mean'), 5)} / {f(g['dp'].get('p1'), 4)} / {f(g['dp'].get('p99'), 4)} | "
                  f"{f(g['kl']['mean'], 5)} / {f(g['kl'].get('p99'), 4)} | {f(g['coverage_top20']['mean'], 4)} | "
                  f"{f(g['margin_at_divergence'].get('p50'), 4)} |")
        # theo loại prompt
        for s in sorted(set(src.values())):
            A = {k: v for k, v in refs[a].items() if src.get(k) == s}
            B = {k: v for k, v in refs[b].items() if src.get(k) == s}
            if A and B:
                js["gen"].setdefault(name + " theo nguồn", {})[s] = gen_closeness(A, B)
    md.append("\nGiống hệt = cả 256 token trùng. Δp = p(token đã chọn) của lần sau − lần đầu, chỉ trên đoạn trước điểm lệch "
              "(cùng ngữ cảnh). KL≈ từ top-20 + phần đuôi (xấp xỉ, xem docs/references E6).\n")

    md.append("## Teacher forcing (200 prompt ngắn + 256 token output tham chiếu)\n")
    if tfs["ref-tf-c1-r1"] and tfs["ref-tf-c8-r1"]:
        t = tf_closeness(tfs["ref-tf-c1-r1"], tfs["ref-tf-c8-r1"])
        js["tf"] = {"c1 vs c8": t}
        md.append("| So sánh | vị trí | same top-1 | Δp mean / RMS / p1 / p99 | KL≈ mean / p99 | độ phủ top-20 |")
        md.append("|---|---|---|---|---|---|")
        md.append(f"| c1 vs c8 | {t['positions']} | {t['same_top1'] * 100:.2f}% ± {t['same_top1_se'] * 196:.2f} | "
                  f"{f(t['dp'].get('mean'), 5)} / {f(t['dp'].get('rms'), 4)} / {f(t['dp'].get('p1'), 4)} / "
                  f"{f(t['dp'].get('p99'), 4)} | {f(t['kl']['mean'], 5)} / {f(t['kl'].get('p99'), 4)} | "
                  f"{f(t['coverage_top20']['mean'], 4)} |\n")

    md.append("## Điểm tác vụ (CI 95% = p ± 1.96·SE, Miller 2024)\n")
    md.append("| Tác vụ | n | Điểm | CI 95% | Ghi chú |")
    md.append("|---|---|---|---|---|")
    js["tasks"] = {}
    fq = score_finqa(run)
    if fq:
        s, per = fq
        js["tasks"]["finqa"] = s
        for t in TOLS:
            a = s[f"acc@{t:g}"]
            md.append(f"| FinQA (RAG), dung sai {t * 100:g}% | {s['n']} | {a['acc'] * 100:.1f} | "
                      f"{a['ci95'][0] * 100:.1f}–{a['ci95'][1] * 100:.1f} | "
                      f"{'**chính**' if t == MAIN_TOL else ''}; không có dòng Answer: {s['no_answer_line']}, "
                      f"bị cắt độ dài: {s['truncated']} |")
        C.write_json(run / "finqa_per_question.json", per)
    for uid, task in (("lm-gsm8k", "gsm8k"), ("lm-ifeval", "ifeval"), ("lm-mmlu-pro", "mmlu_pro")):
        r = lm_results(run, uid, task)
        if not r:
            continue
        js["tasks"][task] = {k: v for k, v in r.items() if k != "per_doc"}
        C.write_json(run / f"{task}_per_doc.json", r["per_doc"])
        for k, v in r.items():
            if isinstance(v, dict) and "acc" in v:
                md.append(f"| {task} {k} | {v['n']} | {v['acc'] * 100:.1f} | {v['ci95'][0] * 100:.1f}–{v['ci95'][1] * 100:.1f} | "
                          f"lm-eval {r.get('lm_eval_version')} |")
    C.write_json(run / "summary.json", js)
    C.atomic_write(run / "summary.md", "\n".join(md) + "\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="thư mục kết quả một lần chạy P06")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "VARIANT"))
    ap.add_argument("--self-flips", nargs=2, metavar=("RUN1", "RUN2"),
                    help="flips tự nhiên: các bài chấm điểm của RUN1 (đơn vị gốc) vs RUN2 (đơn vị -r2), cùng cấu hình")
    a = ap.parse_args()
    if a.self_flips:
        r1, r2 = a.self_flips
        out = {}
        f1, f2 = score_finqa(r1, "finqa-c32-r1"), score_finqa(r2, "finqa-c32-r2")
        if f1 and f2:
            for t in TOLS:
                k = f"ok@{t:g}"
                out[f"finqa@{t:g}"] = paired({q: float(x[k]) for q, x in f1[1].items()},
                                             {q: float(x[k]) for q, x in f2[1].items()})
            out["finqa_pred_changed"] = float(np.mean([f1[1][q]["pred"] != f2[1][q]["pred"] for q in f1[1] if q in f2[1]]))
        for task, u1, u2 in (("gsm8k", "lm-gsm8k", "lm-gsm8k-r2"), ("ifeval", "lm-ifeval", "lm-ifeval-r2"),
                             ("mmlu_pro", "lm-mmlu-pro", "lm-mmlu-pro-r2")):
            x1, x2 = lm_results(r1, u1, task), lm_results(r2, u2, task)
            if x1 and x2:
                for mk in x1["per_doc"]:
                    if mk in x2["per_doc"]:
                        out[f"{task}:{mk}"] = paired(x1["per_doc"][mk], x2["per_doc"][mk])
        C.write_json(Path(r2) / "self_flips.json", out)
        md = ["| Bài (chỉ số) | n | điểm r1 → r2 (hiệu) | paired SE | CI 95% của hiệu | flips đúng→sai / sai→đúng | flips % |",
              "|---|---|---|---|---|---|---|"]
        for k, v in out.items():
            if isinstance(v, dict):
                md.append(f"| {k} | {v['n']} | {v['diff'] * 100:+.2f} | {v['se_paired'] * 100:.2f} | "
                          f"{v['ci95'][0] * 100:+.2f} … {v['ci95'][1] * 100:+.2f} | {v['flips_correct_to_wrong']} / "
                          f"{v['flips_wrong_to_correct']} | {v['flips_pct'] * 100:.2f} |")
        if "finqa_pred_changed" in out:
            md.append(f"\nFinQA: đáp án trích ra khác nhau giữa 2 lần ở {out['finqa_pred_changed'] * 100:.1f}% câu.")
        C.atomic_write(Path(r2) / "self_flips.md", "\n".join(md) + "\n")
        print("\n".join(md))
        return
    if a.compare:
        b, v = a.compare
        out = {"gen": gen_closeness(unit_recs(b, "ref-gen-c8-r1"), unit_recs(v, "ref-gen-c8-r1"))}
        tb, tv = unit_recs(b, "ref-tf-c1-r1"), unit_recs(v, "ref-tf-c1-r1")
        if tb and tv:
            out["tf"] = tf_closeness(tb, tv)
        fb, fv = score_finqa(b), score_finqa(v)
        if fb and fv:
            k = f"ok@{MAIN_TOL:g}"
            out["finqa"] = paired({q: float(x[k]) for q, x in fb[1].items()}, {q: float(x[k]) for q, x in fv[1].items()})
        for uid, task in (("lm-gsm8k", "gsm8k"), ("lm-ifeval", "ifeval"), ("lm-mmlu-pro", "mmlu_pro")):
            rb, rv = lm_results(b, uid, task), lm_results(v, uid, task)
            if rb and rv:
                mk = LM_METRIC[task][0][0]
                out[task] = paired(rb["per_doc"][mk], rv["per_doc"][mk])
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    print(report(a.out))


if __name__ == "__main__":
    main()
