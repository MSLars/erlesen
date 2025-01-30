import spacy


###############################################################################
# Hilfsfunktionen
###############################################################################
def count_syllables(word: str) -> int:
    """
    Schätzt die Anzahl der Silben in einem Wort sehr vereinfacht:
    - Zählt Vokalgruppen
    - Zieht ein 'e' am Ende wieder ab
    """
    word = word.lower()
    syllable_count = 0
    vowels = 'aeiouy'

    if not word:
        return 0

    # Falls erstes Zeichen Vokal ist
    if word[0] in vowels:
        syllable_count += 1

    # Zähle Vokalwechsel
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            syllable_count += 1

    # Häufig enden deutsche Worte auf 'e' (z.B. "Stube"), das oft stumm ist
    # Hier simple Heuristik: Ziehe 1 ab, wenn Wort auf 'e' endet
    if word.endswith('e'):
        syllable_count -= 1

    # Mindestens 1 Silbe
    if syllable_count < 1:
        syllable_count = 1

    return syllable_count


def count_total_syllables(text: str) -> int:
    """
    Zählt die (geschätzten) Silben im gesamten Text.
    """
    words = text.split()
    return sum(count_syllables(word) for word in words)


def is_simple_svo(span) -> bool:
    """
    Prüft heuristisch auf 'Subjekt -> Verb -> (direktes) Objekt' in einem Satz (Span).
    Gibt True zurück, wenn SVO-Abfolge gefunden, sonst False.
    """
    subject_token = None
    verb_token = None
    object_token = None

    tokens = list(span)

    for token in tokens:
        # Subjekt?
        if token.dep_ in ("sb", "nsubj", "nsubj:pass"):
            if subject_token is None:
                subject_token = token
        # Hauptverb (ROOT) - stark vereinfacht
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            if verb_token is None:
                verb_token = token
        # Direktes Objekt (Akkusativobjekt)
        if token.dep_ in ("oa", "obj", "acc"):
            if object_token is None:
                object_token = token

    if subject_token is None or verb_token is None or object_token is None:
        return False

    # Prüfe die Reihenfolge anhand ihres Index im Satz
    s_idx = tokens.index(subject_token)
    v_idx = tokens.index(verb_token)
    o_idx = tokens.index(object_token)

    return (s_idx < v_idx < o_idx)


def analyze_subclauses(span):
    """
    Findet (potentielle) Nebensätze im Span (z.B. durch 'SCONJ', 'oc', 'rc', 'cp').
    Gibt zurück:
      - n_sub (int): Anzahl aller Nebensätze in diesem Satz
      - n_complex (int): Anzahl komplexer Nebensätze in diesem Satz
        (Regel: Ein Satz ist 'komplex', wenn
            a) mehr als 1 Nebensatz-Root
         oder b) genau 1 Root, aber > 8 Tokens in dessen Subtree)
    """
    sub_roots = []
    for token in span:
        if token.dep_ in ("oc", "rc", "cp") or token.pos_ == "SCONJ":
            sub_roots.append(token)

    n_sub = len(sub_roots)
    n_complex = 0

    if n_sub == 0:
        return 0, 0

    # Mehr als 1 Nebensatz-Wurzel => alle sind "komplex"
    if n_sub > 1:
        return n_sub, n_sub

    # Genau 1 Nebensatz
    root = sub_roots[0]
    sub_len = len(list(root.subtree))
    if sub_len > 8:
        return 1, 1  # Komplex
    else:
        return 1, 0  # Einfach


###############################################################################
# Hauptfunktion
###############################################################################
def grammar_evaluation(text: str) -> tuple[float, list[dict]]:
    """
    Untersucht den gegebenen Text hinsichtlich:
      1) Genitiv
      2) Passiv
      3) Konjunktiv
      4) (Lange/komplexe) Nebensätze
      5) SVO-Sätze
      6) Lange Nominalphrasen
      7) Lange Sätze
      8) Lange Wörter (viele Silben)

    Rückgabe:
      (score: float,    # Gesamtbewertung in [0..1]
       verstoss_liste: List[dict])   # alle festgestellten Verstöße

    'verstoss_liste' enthält Elemente wie:
      {
        "type": "genitive",  # oder "passive", "konjunktiv", etc.
        "index": [ (start_char, end_char), ... ]  # Alle Stellen im Text
      }
    """
    # Lade (ggf. nur einmal global) ein deutsches Spacy-Modell
    # Falls du bereits außerhalb ein nlp = spacy.load(...) hast, kannst du das hier ersetzen.
    nlp = spacy.load("de_core_news_sm")
    doc = nlp(text)

    # Hilfswerte
    sents = list(doc.sents)
    sentence_count = len(sents)
    word_count = len(doc)
    syllable_count = count_total_syllables(text)

    # Liste aller Verstöße
    violations = []

    # -----------------------------------------------------------------------
    # 1) Genitiv
    # -----------------------------------------------------------------------
    noun_phrases = list(doc.noun_chunks)
    genitive = [np for np in noun_phrases
                if any("Gen" in t.morph.get("Case") for t in np)]
    # Falls wir Genitivstellen finden, speichern wir sie gemeinsam
    if len(genitive) > 0:
        gen_indices = [(g.start_char, g.end_char) for g in genitive]
        violations.append({
            "type": "genitive",
            "index": gen_indices
        })
    # Score-Anteil: (Anzahl genitive) / (alle NP), mit Obergrenze
    # (Verhältnis, dass "zu viele" Genitive auftauchen)
    susp_max = 0.75 if sentence_count > 2 else 1.0
    n_poss_case = max(1, len(noun_phrases))  # gegen Division durch 0
    tmp_score = min(len(genitive) / n_poss_case, susp_max)
    score_genitive = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 2) Passiv (Hilfsverb "werden"/"sein" + Partizip II "VVPP")
    # -----------------------------------------------------------------------
    passive_pairs = []
    for token in doc:
        # Hilfsverb "werden" oder "sein"?
        if token.lemma_ in ("werden", "sein") and token.pos_ in ("AUX", "VERB"):
            subtree_tokens = list(token.subtree)
            # Partizip II finden
            partizip_2_sub = [t for t in subtree_tokens if t.tag_ == "VVPP"]
            for pp2 in partizip_2_sub:
                passive_pairs.append((token, pp2))
    if len(passive_pairs) > 0:
        # Alle Passivstellen gemeinsam abspeichern
        pass_indices = []
        for (aux, part2) in passive_pairs:
            pass_indices.append((aux.idx, aux.idx + len(aux.text)))
            pass_indices.append((part2.idx, part2.idx + len(part2.text)))
        violations.append({
            "type": "passive",
            "index": pass_indices
        })
    # Score-Anteil
    # n_poss_verb => Anhaltspunkt: Anzahl Verben als "mögliche" Passiv-Kandidaten
    n_poss_verb = len([t for t in doc if t.pos_ == "VERB" or t.pos_ == "AUX"])
    if n_poss_verb == 0:
        n_poss_verb = 1  # Vermeide Division durch 0
    susp_max = 0.5 if sentence_count > 2 else 1.0
    tmp_score = min(len(passive_pairs) / n_poss_verb, susp_max)
    score_passive = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 3) Konjunktiv (Mood=Sub)
    # -----------------------------------------------------------------------
    conj_or_subj = [t for t in doc
                    if t.pos_ in ["VERB", "AUX"] and "Sub" in t.morph.get("Mood")]
    if len(conj_or_subj) > 0:
        conj_indices = [(t.idx, t.idx + len(t.text)) for t in conj_or_subj]
        violations.append({
            "type": "konjunktiv",
            "index": conj_indices
        })
    # Score-Anteil
    susp_max = 0.1 if sentence_count > 2 else 0.5
    tmp_score = min(len(conj_or_subj) / word_count, susp_max) if word_count > 0 else 0
    score_konj = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 4) (Lange und komplexe) Nebensätze
    #    -> wir werten in jedem Satz aus
    # -----------------------------------------------------------------------
    subclause_indices = []
    complex_subclause_indices = []
    total_subclauses = 0
    total_complex_subclauses = 0

    for sent in sents:
        n_sub, n_complex = analyze_subclauses(sent)
        total_subclauses += n_sub
        total_complex_subclauses += n_complex

        # Sammle Indizes: alle Tokens, die Nebensatz-Indikatoren sind
        for token in sent:
            if token.dep_ in ("oc", "rc", "cp") or token.pos_ == "SCONJ":
                # das ist eine Nebensatz-Wurzel
                subclause_indices.append((token.idx, token.idx + len(token.text)))

    # "Normale" Nebensätze
    if total_subclauses > 0:
        violations.append({
            "type": "subclause",
            "index": subclause_indices
        })
    # Komplexe Nebensätze
    # Für diese Demo packen wir sie in denselben Index-Block (man könnte
    # sie aber auch nochmal separat ermitteln)
    if total_complex_subclauses > 0:
        # einfach der Übersicht halber nochmal abgleichen:
        # wer "komplex" ist, hat i.d.R. auch "subclause"-Token
        # Wenn man es auseinander halten will, bräuchte man
        # z.B. das root-Token der komplexen Clause.
        violations.append({
            "type": "complex_subclause",
            "index": subclause_indices
        })

    # Score-Anteil (Heuristik):
    #   - Je mehr (komplexe) Nebensätze, desto höher der Verstoß
    #   - Wir normalisieren auf die Satzanzahl.
    if sentence_count == 0:
        ratio_sub = 0
    else:
        ratio_sub = (total_subclauses + total_complex_subclauses) / sentence_count
    susp_max = 2.0  # z.B. ab 2+ Nebensätzen pro Satz wird "volles" Limit erreicht
    tmp_score = min(ratio_sub, susp_max)
    score_subclauses = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 5) SVO-Fokus: Sätze, die KEIN SVO haben => Verstoß
    # -----------------------------------------------------------------------
    non_svo_indices = []
    svo_count = 0
    for sent in sents:
        if is_simple_svo(sent):
            svo_count += 1
        else:
            # gesamter Satz
            non_svo_indices.append((sent.start_char, sent.end_char))

    if len(non_svo_indices) > 0:
        violations.append({
            "type": "no_svo",
            "index": non_svo_indices
        })
    # Score-Anteil:
    #   Wir wollen möglichst viele SVO-Sätze => je mehr Non-SVO, desto schlechter
    #   (non_svo = sentence_count - svo_count)
    #   ratio = (non_svo / sentence_count)
    if sentence_count == 0:
        ratio_svo = 0
    else:
        ratio_svo = (sentence_count - svo_count) / sentence_count
    # Obergrenze z.B. 1.0 => wenn 100% Sätze ohne SVO, Score = 1
    susp_max = 1.0
    tmp_score = min(ratio_svo, susp_max)
    score_no_svo = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 6) Lange Nominalphrasen (mehr als 3 Tokens)
    # -----------------------------------------------------------------------
    long_nps = [np for np in noun_phrases if len(np) > 3]
    if len(long_nps) > 0:
        np_indices = [(np.start_char, np.end_char) for np in long_nps]
        violations.append({
            "type": "long_noun_phrase",
            "index": np_indices
        })
    # Score-Anteil
    if len(noun_phrases) == 0:
        ratio_nps = 0
    else:
        ratio_nps = len(long_nps) / len(noun_phrases)
    susp_max = 0.3 if sentence_count > 2 else 1.0
    tmp_score = min(ratio_nps, susp_max)
    score_long_np = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 7) Lange Sätze (mehr als 15 Tokens)
    # -----------------------------------------------------------------------
    long_sents = [s for s in sents if len(s) > 15]
    if len(long_sents) > 0:
        long_sent_indices = [(s.start_char, s.end_char) for s in long_sents]
        violations.append({
            "type": "long_sentence",
            "index": long_sent_indices
        })
    # Score-Anteil
    if sentence_count == 0:
        ratio_long_sent = 0
    else:
        ratio_long_sent = len(long_sents) / sentence_count
    susp_max = 3.0  # z.B. bis 3, dann saturiert
    tmp_score = min(ratio_long_sent, susp_max)
    score_long_sents = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # 8) Lange Wörter (mehr als 4 Silben)
    # -----------------------------------------------------------------------
    long_syllable_words = [t for t in doc if count_syllables(t.text) > 4]
    if len(long_syllable_words) > 0:
        lsw_indices = [(w.idx, w.idx + len(w.text)) for w in long_syllable_words]
        violations.append({
            "type": "long_word",
            "index": lsw_indices
        })
    # Score-Anteil
    if word_count == 0:
        ratio_long_words = 0
    else:
        ratio_long_words = len(long_syllable_words) / word_count
    susp_max = 0.2 if sentence_count > 2 else 1.0
    tmp_score = min(ratio_long_words, susp_max)
    score_long_words = tmp_score / susp_max

    # -----------------------------------------------------------------------
    # Kombinierter Score (0..1)
    #   - Teil-Scores sind alle in [0..1], wobei 0 => kein Verstoß, 1 => maximal
    #   - Wir mitteln alle "Verstoßanteile" und ziehen sie von 1 ab
    # -----------------------------------------------------------------------
    partial_scores = [
        score_genitive,
        score_passive,
        score_konj,
        score_subclauses,
        score_no_svo,
        score_long_np,
        score_long_sents,
        score_long_words
    ]
    avg_susp = sum(partial_scores) / len(partial_scores)
    final_score = 1.0 - avg_susp

    # Clamp zwischen 0 und 1
    if final_score < 0:
        final_score = 0.0
    elif final_score > 1:
        final_score = 1.0

    return (final_score, violations)


###############################################################################
# KURZER TEST
###############################################################################
if __name__ == "__main__":
    beispiel_text = "Der Sohn des Metzgers wurde vom Hund gebissen. Hoffe, dass er geheilt werde."
    score, verstosse = grammar_evaluation(beispiel_text)

    print("Score:", score)
    print("Verstöße:")
    for v in verstosse:
        print("  -", v)
