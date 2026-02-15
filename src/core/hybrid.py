def normalize(scores):
    if not scores:
        return {}
    min_s, max_s = min(scores.values()), max(scores.values())
    if min_s == max_s:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


def weighted_hybrid(content, collab, alpha=0.5):
    c = normalize(content)
    cf = normalize(collab)
    items = set(c) | set(cf)

    final = {
        i: alpha * c.get(i, 0) + (1 - alpha) * cf.get(i, 0)
        for i in items
    }
    return sorted(final.items(), key=lambda x: x[1], reverse=True)
