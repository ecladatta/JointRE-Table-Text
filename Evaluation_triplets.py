def evaluation_triplets(pred, gold, relations_possibles):
    exact_tp = 0 # On stocke dans ce variable nombre de vrais positifs pour le match exact
    partial_tp = 0 # On stocke dans ce variable  le nombre de vrais positifs pour le match partiel (sujet+objet)
    partial_rel_tp = 0 # On stocke dans ce variable le nombre de vrais positifs pour le match partiel (relation correcte + une entité correcte)
    total_pred = 0 # On stocke dans ce variable le nombre total de triplets prédits (triplets du participant)
    total_gold = 0 # On stocke dans ce variable le nombre total de triplets annotés (gold)

    # ==================================================
    # Filtrage des triplets prédits : on conserve uniquement les triplets dont la relation appartient à la liste des relations possibles
    # =================================================
    pred = [
        [p for p in pred_doc if p.split(';')[1].strip() in relations_possibles]
        for pred_doc in pred
    ]

    #=====================================================
    # Evaluations
    #======================================================

    for pred_doc, gold_doc in zip(pred, gold): # On parcourt simultanément les triplets prédits et annotés

        total_pred += len(pred_doc) # Ajoute le nombre de triplets du document prédit
        total_gold += len(gold_doc) # Ajoute le nombre de triplets du document annoté

        # --------------------------------------------------
        # Normalisation des triplets
        # --------------------------------------------------
        pred_triples = []
        for p in pred_doc:
            p_parts = [part.strip() for part in p.split(';')] # On split le triplet prédit en trois parties à l'aide du séparateur ';'
            if len(p_parts) == 3:  # Si le triplet est bien constitué de 3 parties (sujet, relation, objet), on le garde
                pred_triples.append(tuple(p_parts))

        gold_triples = []
        for g in gold_doc:
            g_parts = [part.strip() for part in g.split(';')]  # On découpe le triplet annoté (partie gold) en trois parties à l'aide du séparateur ';'
            if len(g_parts) == 3: # Si le triplet est bien constitué de 3 parties, on le conserve
                gold_triples.append(tuple(g_parts))

        # --------------------------------------------------
        # Exact matching (sujet + relation + object) : le triplet prédit doit apparaît exactement parmi les triplets annotés
        # --------------------------------------------------
        for p in pred_triples:
            if p in gold_triples:  # on incrémente le compteur ssi on trouve le triplet prédit dans les triplets annotés
                exact_tp += 1

        # --------------------------------------------------
        # Partial match (sujet + object uniquement): on vérifie uniquement la paire d’entités(sans regarder relation).
        # --------------------------------------------------
        for p in pred_triples:
            p_entities = (p[0], p[2]) # On garde que la paire d'entités (sujet, objet) du triplet prédit
            for g in gold_triples: # Compare avec toutes les paires d'entités du gold
                g_entities = (g[0], g[2])
                if p_entities == g_entities: # Si la paire apparaît, on incrémente le compteur de match partiel
                    partial_tp += 1
                    break # On sort de la boucle pour éviter de le compter deux fois

        # --------------------------------------------------
        # Partial match (relation correcte + une des deux entités correctes)
        # --------------------------------------------------
        for p in pred_triples:
            for g in gold_triples:
                if p[1] == g[1]: # Si la relation est la même
                    if p[0] == g[0] or p[2] == g[2]: # Et au moins un des deux entités correspond
                        partial_rel_tp += 1
                        break # On sort de la boucle pour ne pas compter plusieurs fois

    # --------------------------------------------------
    # Calcul des métriques pour le match exact
    # --------------------------------------------------
    precision_exact = exact_tp / total_pred if total_pred > 0 else 0
    recall_exact = exact_tp / total_gold if total_gold > 0 else 0
    f1_exact = (2 * precision_exact * recall_exact) / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0

    # --------------------------------------------------
    # Calcul des métriques pour le match partiel (sujet + objet)
    # --------------------------------------------------
    precision_partial = partial_tp / total_pred if total_pred > 0 else 0
    recall_partial = partial_tp / total_gold if total_gold > 0 else 0
    f1_partial = (2 * precision_partial * recall_partial) / (precision_partial + recall_partial) if (precision_partial + recall_partial) > 0 else 0

    # --------------------------------------------------
    # Calcul des métriques pour le match partiel (relation correcte + une entité correcte)
    # --------------------------------------------------
    precision_partial_rel = partial_rel_tp / total_pred if total_pred > 0 else 0
    recall_partial_rel = partial_rel_tp / total_gold if total_gold > 0 else 0
    f1_partial_rel = (2 * precision_partial_rel * recall_partial_rel) / (precision_partial_rel + recall_partial_rel) if (precision_partial_rel + recall_partial_rel) > 0 else 0

    # --------------------------------------------------
    # Résultat final
    # --------------------------------------------------
    results = {
        'Exact matching': {'precision': precision_exact, 'recall': recall_exact, 'f1': f1_exact},
        'Partial matching (head+tail)': {'precision': precision_partial, 'recall': recall_partial, 'f1': f1_partial},
        'Partial matching (relation + 1 entity)': {'precision': precision_partial_rel, 'recall': recall_partial_rel, 'f1': f1_partial_rel}
    }
    return results
# Exemple d’utilisation
pred = [
    ["Entho; product_or_service_of; Amara Muzik",
     "Marichika; product_or_service_of; Amara Muzik"],
    ["Charitra; product_or_service_of; Amara Muzik",
     "Bastur Khoje; client_of; Amara Muzik",
    "Bastur Khoje0; product_or_service_of; Amara Muzik0"]
]
gold = [
    ["Entho; product_or_service_of; Amara Muzik",
     "Marichika; product_or_service_of; Amara Muzik"],
    ["Who; product_or_service_of; Amara Muzik",
     "Bastur Khoje; product_or_service_of; Amara Muzik",
    "Bastur Khoje0; product_or_service_of; Amara Muzik000"]
]

relations_possibles= ['acquired_by', 'brand_of','client_of', 'collaboration', 'merged_with', 'product_or_service_of', 'shareholder_of', 'subsidiary_of', 'traded_on']

results = evaluation_triplets(pred, gold, relations_possibles)
for task in results:
    print(f"{task}: {results[task]}")