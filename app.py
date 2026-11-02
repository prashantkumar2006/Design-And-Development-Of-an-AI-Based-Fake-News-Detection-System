import os
import re
import hashlib
import xml.etree.ElementTree as ET
import concurrent.futures
from urllib.parse import urlparse
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# In-memory deterministic cache and analysis history
VERIFICATION_CACHE = {}
ANALYSIS_HISTORY = []

# Load machine learning models and vectorizers
try:
    svm_model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    short_model = joblib.load(os.path.join(MODELS_DIR, "short_text_model.pkl"))
    short_vectorizer = joblib.load(os.path.join(MODELS_DIR, "short_vectorizer.pkl"))
    log_reg_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    naive_bayes_model = joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl"))
    models_loaded = True
    feature_names = tfidf_vectorizer.get_feature_names_out()
except Exception as e:
    print(f"Warning: Could not load model files: {e}")
    models_loaded = False
    feature_names = None

# Persistent HTTP session with connection pooling and retries
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=15, pool_maxsize=15))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

# Pre-warm connection to eliminate first-try cold-start latency
try:
    session.get("https://en.wikipedia.org/w/api.php?action=query&format=json&titles=Main_Page", timeout=3)
    print("Pre-warmed knowledge base connections successfully.")
except Exception:
    pass

# Stopwords for text normalization
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "became", "widely",
    "also", "become", "other", "using", "instruct", "produce", "potential",
    "area", "well", "many", "much", "says", "said", "article", "website",
    "claims", "described", "describes", "provides", "soon", "every", "market",
    "enters", "era", "first", "shows", "reveals", "latest", "today", "news",
    "headline", "title", "claim", "story", "report", "part", "towards", "one",
    "recent", "years"
}

# Reputable journalistic institutions & official scientific/international bodies
TRUSTED_OUTLETS = {
    "bbc", "reuters", "associated press", "ap news", "the hindu", "indian express",
    "times of india", "ndtv", "hindustan times", "isro", "nasa", "cnn", "the guardian",
    "al jazeera", "bloomberg", "forbes", "nature", "scientific american", "pib",
    "washington post", "new york times", "wall street journal", "abc news", "cbs news",
    "nbc news", "time", "pbs", "dw", "france 24", "hindu business line", "livemint",
    "mint", "afp", "ani", "press trust of india", "pti", "spacenews", "space.com",
    "science daily", "the scientist", "the conversation", "cdc", "who", "nih", "lancet",
    "iea", "irena", "eia", "energy agency", "climate council", "cleantechnica"
}

# Fact check markers & fact-checking organizations
FACT_CHECK_MARKERS = [
    "fact check", "fact-check", "factcheck", "snopes", "politifact", "boomlive",
    "false claim", "hoax", "debunk", "misleading", "fake news", "unproven",
    "no evidence", "bogus", "fabricated", "disinformation", "misinformation", "false:", "false "
]

# Extreme clickbait / sensationalist phrases
SENSATIONAL_PATTERNS = [
    r"you won\'?t believe", r"shocking truth", r"what they don\'?t want you to know",
    r"secret cure", r"miracle remedy", r"100% proof", r"bombshell revelation",
    r"wake up sheeple", r"mainstream media won\'?t show", r"spread this before it\'?s deleted"
]

# Explicit red flags inside the text where the article itself admits lack of evidence
SKEPTIC_PATTERNS = [
    r"provides? no (?:manufacturer )?(?:announcement|evidence|proof|laboratory results|independent testing)",
    r"no (?:scientific|independent|verifiable) (?:evidence|testing|proof|results)",
    r"without (?:any )?(?:proof|evidence|testing|verification)",
    r"unsubstantiated claim",
    r"has not been (?:verified|independently tested|proven)",
    r"claims? without (?:evidence|proof)",
    r"no manufacturer announcement"
]

ACTION_VERBS = {
    "endorses", "endorsed", "endorsing", "endorsement", "arrested", "arrest", "resigns", "resigned",
    "lands", "landed", "landing", "cures", "cured", "cure", "dies", "died", "killed", "cancels",
    "bans", "banned", "approves", "approved", "wins", "won", "steals", "stolen",
    "develops", "developed", "development", "confirms", "confirmed", "admits", "admitted"
}

HIGH_IMPACT_ACTION_CLAIMS = {
    "endorses", "endorsed", "endorsing", "endorsement", "arrested", "arrest",
    "resigns", "resigned", "indicted", "assassinated", "executed", "banned"
}

PREFIX_PATTERN = r"^(?:headline|title|news|breaking|claim|article|story|report)\s*:\s*"

def normalize_text(text):
    text = str(text).replace("\xa0", " ").replace("’", "'").replace("“", '"').replace("”", '"').replace("—", " - ").replace("–", " - ")
    text = re.sub(PREFIX_PATTERN, "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def check_internal_skepticism(text):
    for p in SKEPTIC_PATTERNS:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return True, match.group(0)
    return False, None

def informative_tokens(s):
    tokens = set(re.findall(r"\b[a-z0-9\-]{3,}\b", s.lower()))
    return {t.strip("-") for t in tokens if t.strip("-") not in STOPWORDS and len(t.strip("-")) > 2}

def extract_smart_query(text):
    entities = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b", text)
    words = [w for w in re.findall(r"\b[a-zA-Z0-9\-]{3,}\b", text) if w.lower() not in STOPWORDS]
    action_words = [w for w in re.findall(r"\b[a-zA-Z\-]+\b", text) if w.lower() in ACTION_VERBS]
    
    ordered_terms = []
    seen = set()
    for act in action_words:
        if act.lower() not in seen:
            ordered_terms.append(act)
            seen.add(act.lower())
    for ent in entities:
        for part in ent.split():
            clean_part = part.strip("-")
            if clean_part.lower() not in STOPWORDS and clean_part.lower() not in seen and len(clean_part) > 2:
                ordered_terms.append(clean_part)
                seen.add(clean_part.lower())
    for w in words:
        clean_w = w.strip("-")
        if clean_w.lower() not in seen and len(clean_w) > 2:
            ordered_terms.append(clean_w)
            seen.add(clean_w.lower())
    return " ".join(ordered_terms[:7])

def fetch_live_news(query):
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = session.get(url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            items = []
            for it in root.findall(".//item")[:12]:
                title = it.find("title").text if it.find("title") is not None else ""
                link = it.find("link").text if it.find("link") is not None else ""
                src_el = it.find("source")
                src = src_el.text if src_el is not None else "News Source"
                items.append({"title": title, "source": src, "link": link})
            return items
    except Exception as e:
        print(f"Live news query notice: {e}")
    return []

def fetch_wikipedia(query, full_text):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        resp = session.get(url, params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3
        }, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("query", {}).get("search", [])
            text_tokens = informative_tokens(full_text)
            text_actions = {w.lower() for w in re.findall(r"\b[a-zA-Z\-]+\b", full_text) if w.lower() in HIGH_IMPACT_ACTION_CLAIMS}
            
            matches = []
            for item in items:
                title = item.get("title", "")
                snippet = re.sub(r"<.*?>", "", item.get("snippet", ""))
                wiki_tokens = informative_tokens(title + " " + snippet)
                common = text_tokens.intersection(wiki_tokens)
                
                if text_actions:
                    wiki_all_words = {w.lower() for w in re.findall(r"\b[a-zA-Z\-]+\b", title + " " + snippet)}
                    action_confirmed = any(act[:4] in w for act in text_actions for w in wiki_all_words)
                    if not action_confirmed:
                        continue

                if len(common) >= 4:
                    matches.append({
                        "title": title,
                        "source": "Wikipedia (Verified Knowledge Base)",
                        "snippet": snippet,
                        "link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "common_count": len(common)
                    })
            return matches
    except Exception as e:
        print(f"Wikipedia query notice: {e}")
    return []

def clean_text_for_svm(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>|http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in STOPWORDS)

def compute_linguistic_signals(text):
    """Computes stylistic, sensationalism, and credibility markers from article text."""
    if not text:
        return {
            "sensationalism_score": 10,
            "sensationalism_level": "Low",
            "sensational_words": [],
            "all_caps_count": 0,
            "all_caps_ratio": 0.0,
            "exclamation_count": 0,
            "readability_label": "Standard Journalistic",
            "readability_score": "College / News Standard",
            "tone_label": "Objective / Third-Person",
            "word_count": 0
        }

    words = re.findall(r"\b[A-Za-z]+\b", text)
    word_count = len(words)
    exclamations = text.count("!")

    # 1. All-Caps detection (words with >= 2 characters, excluding common acronyms)
    COMMON_ACRONYMS = {"NASA", "ISRO", "WHO", "CDC", "G20", "RNA", "DNA", "COVID", "AI", "US", "USA", "UK", "EU", "UN", "PTI", "ANI", "BBC", "CNN", "SVM", "ML"}
    caps_words = [w for w in words if w.isupper() and len(w) >= 2 and w not in COMMON_ACRONYMS]
    caps_ratio = round(len(caps_words) / max(word_count, 1), 3)

    # 2. Sensational buzzwords
    SENSATIONAL_BUZZWORDS = {
        "shocking", "unbelievable", "secret", "miracle", "exposed", "conspiracy",
        "cure", "cures", "scandal", "urgent", "warning", "explosive", "bombshell",
        "hidden", "banned", "hoax", "bizarre", "jawdropping", "truth", "insane",
        "breakthrough", "mindblowing", "leaked", "coverup", "panic", "sheeple"
    }
    lower_words = [w.lower() for w in words]
    found_buzzwords = list(set(w for w in lower_words if w in SENSATIONAL_BUZZWORDS))

    # Calculate sensationalism score (0 to 100)
    score = 8
    score += len(found_buzzwords) * 20
    score += min(exclamations * 8, 24)
    if caps_ratio > 0.04:
        score += min(int(caps_ratio * 120), 25)
    score = max(5, min(score, 98))

    if score >= 60:
        level = "High"
    elif score >= 28:
        level = "Moderate"
    else:
        level = "Low"

    # 3. Readability & Sentence Complexity
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sentence_count
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

    if avg_sentence_len > 24 or avg_word_len > 5.6:
        readability_label = "Investigative / Formal"
        readability_score = "Advanced / In-Depth Report"
    elif avg_sentence_len >= 13:
        readability_label = "Standard Journalistic"
        readability_score = "News Wire Standard"
    else:
        readability_label = "Conversational / Casual"
        readability_score = "Fast-Paced / Short Sentences"

    # 4. Subjectivity / Emotional Tone
    first_second_pronouns = {"i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours"}
    pronoun_count = sum(1 for w in lower_words if w in first_second_pronouns)
    pronoun_ratio = pronoun_count / max(word_count, 1)

    if pronoun_ratio > 0.04 or score >= 60:
        tone_label = "Subjective / Opinionated Tone"
    elif pronoun_ratio > 0.02 or score >= 35:
        tone_label = "Informative / Mixed Tone"
    else:
        tone_label = "Objective / Neutral Journalistic"

    return {
        "sensationalism_score": score,
        "sensationalism_level": level,
        "sensational_words": found_buzzwords[:5],
        "all_caps_count": len(caps_words),
        "all_caps_ratio": caps_ratio,
        "exclamation_count": exclamations,
        "readability_label": readability_label,
        "readability_score": readability_score,
        "tone_label": tone_label,
        "word_count": word_count
    }

def get_explainable_ai_data(text):
    """Computes Explainable AI feature attributions and multi-classifier comparison."""
    if not models_loaded or not feature_names is not None:
        return {"features": [], "comparison": {}}

    try:
        cleaned = clean_text_for_svm(text)
        if not cleaned:
            return {"features": [], "comparison": {}}

        feats = tfidf_vectorizer.transform([cleaned])
        non_zero = feats.nonzero()[1]

        # 1. Feature Importance Attributions
        weights = []
        for idx in non_zero:
            coef = float(svm_model.coef_[0][idx])
            val = float(feats[0, idx])
            impact = coef * val
            fn = feature_names[idx].replace("word__", "").replace("char__", "[char] ")
            weights.append({
                "feature": fn,
                "impact": round(impact, 4),
                "direction": "REAL" if coef > 0 else "FAKE",
                "weight": round(coef, 4)
            })

        weights.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_features = weights[:7]

        # 2. Multi-Classifier Comparison
        svm_score = float(svm_model.decision_function(feats)[0])
        svm_pred = "REAL" if svm_score > 0 else "FAKE"

        lr_pred_raw = int(log_reg_model.predict(feats)[0])
        lr_prob_real = float(log_reg_model.predict_proba(feats)[0][1])
        lr_pred = "REAL" if lr_pred_raw == 1 else "FAKE"

        nb_pred_raw = int(naive_bayes_model.predict(feats)[0])
        nb_prob_real = float(naive_bayes_model.predict_proba(feats)[0][1])
        nb_pred = "REAL" if nb_pred_raw == 1 else "FAKE"

        comparison = {
            "svm": {
                "name": "Linear SVM",
                "prediction": svm_pred,
                "confidence": round(min(abs(svm_score) * 40 + 50, 99.8), 1),
                "margin_score": round(svm_score, 3)
            },
            "logistic_regression": {
                "name": "Logistic Regression",
                "prediction": lr_pred,
                "confidence": round(lr_prob_real * 100 if lr_pred == "REAL" else (1 - lr_prob_real) * 100, 1),
                "real_probability": round(lr_prob_real, 3)
            },
            "naive_bayes": {
                "name": "Naive Bayes",
                "prediction": nb_pred,
                "confidence": round(nb_prob_real * 100 if nb_pred == "REAL" else (1 - nb_prob_real) * 100, 1),
                "real_probability": round(nb_prob_real, 3)
            }
        }

        return {
            "features": top_features,
            "comparison": comparison
        }
    except Exception as e:
        print(f"Explainable AI calculation notice: {e}")
        return {"features": [], "comparison": {}}

def verify_hybrid(raw_text):
    text = normalize_text(raw_text)
    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if cache_key in VERIFICATION_CACHE:
        return VERIFICATION_CACHE[cache_key]

    # Explainable AI & Comparison computation
    xai_data = get_explainable_ai_data(text)
    linguistic_signals = compute_linguistic_signals(raw_text)

    def finalize(result_dict):
        result_dict["linguistic_signals"] = linguistic_signals
        if "explainable_ai" not in result_dict:
            result_dict["explainable_ai"] = xai_data
        VERIFICATION_CACHE[cache_key] = result_dict
        return result_dict

    # 1. Internal Evidence & Skepticism Check
    has_skepticism, reason = check_internal_skepticism(text)
    if has_skepticism:
        res = {
            "verdict": "fake",
            "label": "UNVERIFIED / UNSUBSTANTIATED CLAIM",
            "model": "Evidence & Consistency Verification",
            "interpretation": "Explicit Lack of Empirical Evidence",
            "description": f'The article itself explicitly acknowledges that it "{reason}", meaning the claim is unproven and lacks laboratory, testing, or manufacturer support.',
            "sources": [],
            "note": "Articles claiming breakthroughs without manufacturer announcements or verifiable testing are speculative or fabricated.",
            "explainable_ai": xai_data
        }
        return finalize(res)

    # 2. Extract robust search query
    query = extract_smart_query(text)
    claim_tokens = informative_tokens(query)

    # 3. Parallel live web search (Google News RSS + Wikipedia API)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_news = executor.submit(fetch_live_news, query)
        f_wiki = executor.submit(fetch_wikipedia, query, text)
        news_items = f_news.result()
        wiki_matches = f_wiki.result()

    claim_actions = {w.lower() for w in re.findall(r"\b[a-zA-Z\-]+\b", text) if w.lower() in HIGH_IMPACT_ACTION_CLAIMS}

    # A. Check for Fact-Check / Debunking & Live Press Corroboration
    debunk_matches = []
    corroborated = []
    for item in news_items:
        t_lower = item["title"].lower()
        s_lower = item["source"].lower()
        cand_tokens = informative_tokens(item["title"])
        common = claim_tokens.intersection(cand_tokens)
        coverage = len(common) / max(len(claim_tokens), 1)

        is_fact_check = any(marker in t_lower for marker in FACT_CHECK_MARKERS) or any(marker in s_lower for marker in FACT_CHECK_MARKERS)
        if is_fact_check:
            # Only count as a debunk if it directly investigates the core tokens of this claim
            if (coverage >= 0.40 and len(common) >= 3) or coverage >= 0.50:
                debunk_matches.append(item)
            continue

        if claim_actions and not any(act in t_lower for act in claim_actions):
            # Cannot corroborate a specific high-impact action (like endorsement) without the action itself
            continue

        is_trusted = any(out in s_lower or out in t_lower for out in TRUSTED_OUTLETS)
        if is_trusted and coverage >= 0.55:
            corroborated.append(item)
        elif coverage >= 0.70:
            corroborated.append(item)

    if debunk_matches:
        res = {
            "verdict": "fake",
            "label": "DEBUNKED / FALSE CLAIM",
            "model": "Live Fact-Check Intelligence",
            "interpretation": "Identified as False by Fact-Checking Media",
            "description": "Fact-checking organizations have investigated and refuted this specific claim.",
            "sources": debunk_matches[:4],
            "note": "Independent verification shows this claim has been investigated and flagged as false or fabricated.",
            "explainable_ai": xai_data
        }
        return finalize(res)

    if corroborated:
        first_source = corroborated[0]["source"]
        res = {
            "verdict": "real",
            "label": "VERIFIED REAL NEWS",
            "model": "Live Multi-Source Corroboration",
            "interpretation": "Confirmed by Verified Press Outlets",
            "description": f"This event is actively reported and confirmed by established media outlets, including {first_source}.",
            "sources": corroborated[:4],
            "note": "Corroborated across live news sources. Multiple independent publishers have covered this event.",
            "explainable_ai": xai_data
        }
        return finalize(res)

    # B. Check Scientific & Knowledge Base (Wikipedia Official API)
    FICTION_MARKERS = [
        "fiction", "fictional", "myth", "mythology", "folklore", "hoax",
        "urban legend", "conspiracy theory", "satire", "parody", "proverb"
    ]
    if wiki_matches and wiki_matches[0]["common_count"] >= 4:
        top_w = wiki_matches[0]
        is_fiction = any(fm in top_w["title"].lower() or fm in top_w["snippet"].lower() for fm in FICTION_MARKERS)
        if is_fiction:
            res = {
                "verdict": "fake",
                "label": "FICTION / MYTH / FABRICATED CLAIM",
                "model": "Encyclopedic Myth & Fiction Index",
                "interpretation": "Documented in Archives as Myth or Fiction",
                "description": f"This claim is documented in encyclopedic archives as folklore, fiction, or an urban myth: {top_w['title']}.",
                "sources": [
                    {"title": top_w["title"] + " — " + top_w["snippet"][:120] + "...", "source": "Wikipedia", "link": top_w["link"]}
                ],
                "note": "Documented as folklore, myth, or fiction rather than an empirical scientific event.",
                "explainable_ai": xai_data
            }
            return finalize(res)

        res = {
            "verdict": "real",
            "label": "VERIFIED SCIENTIFIC / FACTUAL EVENT",
            "model": "Knowledge Base Verification (Wikipedia)",
            "interpretation": "Authoritative Encyclopedic & Scientific Record",
            "description": f"This development/concept is documented in verified medical and scientific records: {top_w['title']}.",
            "sources": [
                {"title": top_w["title"] + " — " + top_w["snippet"][:120] + "...", "source": "Wikipedia", "link": top_w["link"]}
            ],
            "note": "Documented in established encyclopedic archives. The underlying technology and institutions are verified.",
            "explainable_ai": xai_data
        }
        return finalize(res)

    # 4. Check Clickbait & Sensationalism Heuristics
    sensational_hits = [p for p in SENSATIONAL_PATTERNS if re.search(p, text.lower())]
    all_caps_chars = len(re.findall(r"[A-Z]", text))
    all_caps_ratio = all_caps_chars / (len(text) + 1)
    if sensational_hits or all_caps_ratio > 0.40:
        res = {
            "verdict": "fake",
            "label": "UNVERIFIED / SENSATIONALIST",
            "model": "Stylistic & Heuristic Filter",
            "interpretation": "Sensationalist Style & Zero Verified Sources",
            "description": "Text exhibits intense emotional manipulation / clickbait markers and has zero corroboration from reputable news publishers.",
            "sources": [],
            "note": "No credible news publishers have reported this claim, and the writing style matches known viral disinformation patterns.",
            "explainable_ai": xai_data
        }
        return finalize(res)

    # 5. Fallback to Machine Learning (Linear SVM + TF-IDF)
    cleaned = clean_text_for_svm(text)
    if cleaned and models_loaded:
        word_count = len(cleaned.split())
        if word_count <= 25 and short_model is not None and short_vectorizer is not None:
            features = short_vectorizer.transform([cleaned])
            pred = short_model.predict(features)[0]
            model_name = "Linear SVM (Headline Classifier)"
        else:
            features = tfidf_vectorizer.transform([cleaned])
            pred = svm_model.predict(features)[0]
            model_name = "Linear SVM (Article Classifier)"

        verdict = "real" if int(pred) == 1 else "fake"
        label = "REAL NEWS (ML SIGNAL)" if verdict == "real" else "FAKE / MISLEADING (ML SIGNAL)"

        res = {
            "verdict": verdict,
            "label": label,
            "model": model_name,
            "interpretation": "Statistical Text-Pattern Match",
            "description": (
                "The machine learning classifier matched this text structure and vocabulary "
                "with verified journalistic writing patterns." if verdict == "real" else
                "The machine learning classifier identified linguistic patterns frequently found in unverified or misleading articles."
            ),
            "sources": [],
            "note": "Evaluated using the trained Linear SVM model on TF-IDF textual features.",
            "explainable_ai": xai_data
        }
        return finalize(res)

    # Default fallback
    res = {
        "verdict": "unknown",
        "label": "UNVERIFIED CLAIM",
        "model": "TruthLens Analysis Engine",
        "interpretation": "Insufficient Evidence",
        "description": "Could not locate matching reports from trusted news networks or clear linguistic markers.",
        "sources": [],
        "note": "Please verify this story against official publications before sharing.",
        "explainable_ai": xai_data
    }
    return finalize(res)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "ready": True,
        "model": "Multi-Layer Engine (Knowledge Base + Live Corroboration + Linear SVM)",
        "classifiers": ["Linear SVM (99.63%)", "Logistic Regression (99.28%)", "Naive Bayes (94.63%)"]
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please paste a headline or article text to analyze."}), 400

    result = verify_hybrid(text)

    # Save to in-memory analysis history
    snippet = text.split("\n")[0][:90]
    history_entry = {
        "id": hashlib.md5((text + str(len(ANALYSIS_HISTORY))).encode()).hexdigest()[:8],
        "title": snippet if snippet else "Untitled Article",
        "text": text,
        "verdict": result["verdict"],
        "label": result["label"],
        "model": result["model"],
        "interpretation": result.get("interpretation", "Evidence Cross-Check"),
        "description": result.get("description", ""),
        "sources": result.get("sources", []),
        "sources_count": len(result.get("sources", [])),
        "note": result.get("note", ""),
        "linguistic_signals": result.get("linguistic_signals", {}),
        "explainable_ai": result.get("explainable_ai", {})
    }
    # Keep last 20 entries
    ANALYSIS_HISTORY.insert(0, history_entry)
    if len(ANALYSIS_HISTORY) > 20:
        ANALYSIS_HISTORY.pop()

    return jsonify(result)

@app.route("/api/extract-url", methods=["POST"])
def extract_url():
    """Extracts headline and article text from a provided web URL using BeautifulSoup."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "Please provide a valid URL starting with http:// or https://"}), 400

    try:
        resp = session.get(url, timeout=7)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        # 1. Extract Title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # 2. Decompose navigational, script, footer elements
        for junk in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
            junk.decompose()

        # 3. Extract meaningful article paragraphs
        paragraphs = []
        for p in soup.find_all("p"):
            p_text = p.get_text(strip=True)
            if len(p_text.split()) >= 7:
                paragraphs.append(p_text)

        body_text = "\n\n".join(paragraphs[:30])
        if not body_text:
            body_text = soup.get_text(separator=" ", strip=True)[:4000]

        if not body_text or len(body_text.split()) < 10:
            return jsonify({"error": "Could not extract sufficient article text from this web page. Please copy and paste the text manually."}), 400

        domain = urlparse(url).netloc.replace("www.", "")
        full_text = f"{title}\n\n{body_text}" if title and not body_text.startswith(title) else body_text

        return jsonify({
            "success": True,
            "url": url,
            "domain": domain,
            "title": title or "Extracted Article",
            "text": full_text.strip(),
            "word_count": len(full_text.split())
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve article: {str(e)}"}), 400

@app.route("/api/batch", methods=["POST"])
def batch_predict():
    data = request.get_json(silent=True) or {}
    articles = data.get("articles", [])

    if not articles or not isinstance(articles, list):
        return jsonify({"error": "Please provide an array of article texts under 'articles'."}), 400

    # Max 10 articles per batch to ensure fast response
    articles = [a.strip() for a in articles[:10] if isinstance(a, str) and a.strip()]
    if not articles:
        return jsonify({"error": "No valid text items found in batch."}), 400

    results = []
    real_count = 0
    fake_count = 0
    unverified_count = 0

    for idx, text in enumerate(articles, 1):
        res = verify_hybrid(text)
        snippet = text.split("\n")[0][:100]
        verdict = res["verdict"]

        if verdict == "real":
            real_count += 1
        elif verdict == "fake":
            fake_count += 1
        else:
            unverified_count += 1

        results.append({
            "index": idx,
            "snippet": snippet,
            "verdict": verdict,
            "label": res["label"],
            "model": res["model"],
            "sources": res.get("sources", [])[:2]
        })

    summary = {
        "total": len(results),
        "real": real_count,
        "fake": fake_count,
        "unverified": unverified_count,
        "credibility_score": round((real_count / len(results)) * 100, 1)
    }

    return jsonify({
        "summary": summary,
        "results": results
    })

@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"history": ANALYSIS_HISTORY})

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    ANALYSIS_HISTORY.clear()
    return jsonify({"success": True, "message": "Analysis history cleared."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
