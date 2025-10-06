def evaluation_triplets(pred, gold, relations_possibles):
    import re

    def normalize_entity(e):
        return re.sub(r'\s+', '', e.lower())
    def entities_equal(e1, e2):
        e1, e2 = normalize_entity(e1), normalize_entity(e2)
        return e1 == e2 or e1 in e2 or e2 in e1

    def triples_equal(t1, t2):
        h1, r1, t1_ = t1
        h2, r2, t2_ = t2

        if r1 != r2:
            return False

        if entities_equal(h1, h2) and entities_equal(t1_, t2_):
            return True
        if entities_equal(h1, t2_) and entities_equal(t1_, h2):
            return True

        return False

    def entities_pair_equal(p1, p2):
        h1, t1_ = p1
        h2, t2_ = p2

        if entities_equal(h1, h2) and entities_equal(t1_, t2_):
            return True
        if entities_equal(h1, t2_) and entities_equal(t1_, h2):
            return True
        return False

    exact_tp = 0
    partial_tp = 0
    partial_rel_tp = 0
    total_pred = 0
    total_gold = 0


    pred = [
        [p for p in pred_doc if p.split(';')[1].strip() in relations_possibles]
        for pred_doc in pred
    ]

    for pred_doc, gold_doc in zip(pred, gold):
        total_pred += len(pred_doc)
        total_gold += len(gold_doc)

        pred_triples = []
        for p in pred_doc:
            p_parts = [part.strip() for part in p.split(';')]
            if len(p_parts) == 3:
                pred_triples.append(tuple(p_parts))

        gold_triples = []
        for g in gold_doc:
            g_parts = [part.strip() for part in g.split(';')]
            if len(g_parts) == 3:
                gold_triples.append(tuple(g_parts))

        for p in pred_triples:
            if any(triples_equal(p, g) for g in gold_triples):
                exact_tp += 1

        for p in pred_triples:
            p_entities = (p[0], p[2])
            if any(entities_pair_equal(p_entities, (g[0], g[2])) for g in gold_triples):
                partial_tp += 1

        for p in pred_triples:
            for g in gold_triples:
                if p[1] == g[1]: 
                    if entities_equal(p[0], g[0]) or entities_equal(p[2], g[2]) \
                       or entities_equal(p[0], g[2]) or entities_equal(p[2], g[0]):
                        partial_rel_tp += 1
                        break

    # Métriques Exact
    precision_exact = exact_tp / total_pred if total_pred > 0 else 0
    recall_exact = exact_tp / total_gold if total_gold > 0 else 0
    f1_exact = (2 * precision_exact * recall_exact) / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0

    # Métriques Partial (head+tail)
    precision_partial = partial_tp / total_pred if total_pred > 0 else 0
    recall_partial = partial_tp / total_gold if total_gold > 0 else 0
    f1_partial = (2 * precision_partial * recall_partial) / (precision_partial + recall_partial) if (precision_partial + recall_partial) > 0 else 0

    # Métriques Partial (relation + une entité correcte)
    precision_partial_rel = partial_rel_tp / total_pred if total_pred > 0 else 0
    recall_partial_rel = partial_rel_tp / total_gold if total_gold > 0 else 0
    f1_partial_rel = (2 * precision_partial_rel * recall_partial_rel) / (precision_partial_rel + recall_partial_rel) if (precision_partial_rel + recall_partial_rel) > 0 else 0

    results = {
        'Exact matching': {'precision': precision_exact, 'recall': recall_exact, 'f1': f1_exact},
        'Partial matching (head+tail)': {'precision': precision_partial, 'recall': recall_partial, 'f1': f1_partial},
        'Partial matching (relation + 1 entity)': {'precision': precision_partial_rel, 'recall': recall_partial_rel, 'f1': f1_partial_rel}
    }
    return results