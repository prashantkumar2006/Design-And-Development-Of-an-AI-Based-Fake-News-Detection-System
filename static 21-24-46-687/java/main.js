// DOM Elements
const form = document.querySelector('#analyze-form');
const textarea = document.querySelector('#news-text');
const count = document.querySelector('#char-count');
const button = document.querySelector('#analyze-button');
const errorBox = document.querySelector('#form-error');
const resultPanel = document.querySelector('#result-panel');
const status = document.querySelector('#backend-status');
const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('#nav');

const resultTitle = document.querySelector('#result-title');
const resultTag = document.querySelector('#result-tag');
const resultDesc = document.querySelector('#result-description');
const resultModel = document.querySelector('#result-model');
const resultInterp = document.querySelector('#result-interpretation');
const resultNote = document.querySelector('#result-note');
const sourcesBlock = document.querySelector('#result-sources-block');
const sourcesList = document.querySelector('#result-sources-list');

// URL Extractor & Input Mode elements
const tabModeText = document.querySelector('#tab-mode-text');
const tabModeUrl = document.querySelector('#tab-mode-url');
const urlInputBlock = document.querySelector('#url-input-block');
const articleUrlInput = document.querySelector('#article-url-input');
const extractUrlBtn = document.querySelector('#extract-url-btn');
const urlStatus = document.querySelector('#url-status');
const exportReportBtn = document.querySelector('#export-report-btn');

// Linguistic Radar elements
const radarBlock = document.querySelector('#radar-block');
const radarSensBadge = document.querySelector('#radar-sens-badge');
const radarSensFill = document.querySelector('#radar-sens-fill');
const radarSensDesc = document.querySelector('#radar-sens-desc');
const radarToneBadge = document.querySelector('#radar-tone-badge');
const radarToneDesc = document.querySelector('#radar-tone-desc');
const radarTonePronouns = document.querySelector('#radar-tone-pronouns');
const radarCapsBadge = document.querySelector('#radar-caps-badge');
const radarUrgencyDetail = document.querySelector('#radar-urgency-detail');
const radarReadBadge = document.querySelector('#radar-read-badge');
const radarReadLabel = document.querySelector('#radar-read-label');
const radarReadDesc = document.querySelector('#radar-read-desc');

// Explainable AI & Comparison elements
const xaiBlock = document.querySelector('#xai-block');
const xaiBars = document.querySelector('#xai-bars');
const toggleHeatmapBtn = document.querySelector('#toggle-heatmap-btn');
const xaiHeatmapWrap = document.querySelector('#xai-heatmap-wrap');
const comparisonBlock = document.querySelector('#comparison-block');
const compSvmPred = document.querySelector('#comp-svm-pred');
const compSvmConf = document.querySelector('#comp-svm-conf');
const compLrPred = document.querySelector('#comp-lr-pred');
const compLrConf = document.querySelector('#comp-lr-conf');
const compNbPred = document.querySelector('#comp-nb-pred');
const compNbConf = document.querySelector('#comp-nb-conf');

let currentXaiFeatures = [];
let currentRawArticle = '';

// History elements
const historyList = document.querySelector('#history-list');
const historyEmpty = document.querySelector('#history-empty');
const clearHistoryBtn = document.querySelector('#clear-history-btn');

// Batch elements
const batchText = document.querySelector('#batch-text');
const batchSubmitBtn = document.querySelector('#batch-submit-btn');
const batchResultsWrap = document.querySelector('#batch-results-wrap');
const batchTableBody = document.querySelector('#batch-table-body');
const batchTotal = document.querySelector('#batch-total');
const batchReal = document.querySelector('#batch-real');
const batchFake = document.querySelector('#batch-fake');
const batchUnverified = document.querySelector('#batch-unverified');
const batchScore = document.querySelector('#batch-score');
const loadBatchSampleBtn = document.querySelector('#load-batch-sample');

// Interactive How TruthLens Works steps data
const INTERACTIVE_STEPS = {
    1: {
        badge: "STAGE 01 OF 04",
        title: "Stage 1: Entity & Predicate Normalization",
        desc: "Raw input is cleaned of Unicode artifacts, format labels (e.g. 'Headline:', 'Breaking:'), and non-breaking spaces. High-signal named entities, proper nouns (e.g. 'Chandrayaan-3', 'Moderna', 'Pfizer'), and action predicates are extracted to construct focused search queries.",
        code: `Input: "Headline: India's Chandrayaan-3 Successfully Lands on the Moon..."<br>
➔ Cleaned Query: <b>[Chandrayaan-3, Moon, Lands, Vikram, ISRO]</b>`
    },
    2: {
        badge: "STAGE 02 OF 04",
        title: "Stage 2: Real-Time Fact-Check & Knowledge Base Cross-Check",
        desc: "The engine queries live press feeds (Google News RSS) and authoritative encyclopedias (Wikipedia API) concurrently via connection-pooled HTTP sessions. It cross-checks whether accredited global publishers or scientific registries confirm the reported event.",
        code: `Google News RSS: <b>Matched [PBS, Nature, BBC]</b> (Coverage: 83.3%)<br>
Wikipedia API: <b>Confirmed Entity [Chandrayaan-3]</b> (4 Shared Concepts)`
    },
    3: {
        badge: "STAGE 03 OF 04",
        title: "Stage 3: Multi-Classifier Ensemble Evaluation",
        desc: "The text is transformed into 100,000 word and character n-gram TF-IDF feature unions. Predictions are computed simultaneously across Linear SVM (Primary, 99.6% Accuracy), Logistic Regression (99.28%), and Multinomial Naive Bayes (94.63%).",
        code: `Linear SVM: <b>REAL (Margin Score: +1.42)</b><br>
Logistic Regression: <b>REAL (98.4% Confidence)</b> | Naive Bayes: <b>REAL (96.1%)</b>`
    },
    4: {
        badge: "STAGE 04 OF 04",
        title: "Stage 4: Explainable AI (XAI) Attribution & Synthesis",
        desc: "To eliminate black-box opacity, the system calculates exact mathematical attributions (TF-IDF Weight × SVM Coefficient) for every active token. It highlights which words drove the verdict towards 'Real' versus 'Fake' and synthesizes citations.",
        code: `Token Impact: <b>'lunar' (+0.14)</b>, <b>'mission' (+0.12)</b>, <b>'lander' (+0.11)</b><br>
Verdict Synthesis: <b>VERIFIED REAL NEWS (Corroborated by Verified Press)</b>`
    }
};

// Step Tab Interaction
document.querySelectorAll('.step-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const step = tab.getAttribute('data-step');
        const data = INTERACTIVE_STEPS[step];
        if (data) {
            document.querySelector('#step-badge').textContent = data.badge;
            document.querySelector('#step-title').textContent = data.title;
            document.querySelector('#step-desc').textContent = data.desc;
            document.querySelector('#step-visual').innerHTML = `<div class="code-box"><code>${data.code}</code></div>`;
        }
    });
});

// Sample dataset for instant one-click testing
const SAMPLES = {
    chandrayaan: "India's Chandrayaan-3 Successfully Lands on the Moon\n\nIndia's Chandrayaan-3 mission successfully landed near the Moon's south polar region on 23 August 2023. The Vikram lander completed its soft landing, and the Pragyan rover was subsequently deployed to explore the lunar surface.",
    solar: "Solar Power Becomes One of the World's Fastest-Growing Electricity Sources\n\nSolar energy capacity has expanded significantly in recent years as countries invest in renewable electricity generation. Improvements in photovoltaic technology, manufacturing and installation have contributed to lower solar power costs.",
    mrna: "Scientists Develop a Vaccine Using Messenger RNA Technology\n\nMessenger RNA technology became widely recognised during the COVID-19 pandemic through vaccines developed by Pfizer-BioNTech and Moderna. These vaccines instruct cells to produce a harmless protein that triggers an immune response.",
    battery: "New Smartphone Battery Technology Allows One-Year Charging\n\nA technology website claims that a newly developed smartphone battery can operate continuously for one year without requiring a recharge. The article describes the technology as a breakthrough that will soon be available in every smartphone. However, it provides no manufacturer announcement, laboratory results or independent testing to support the claim.",
    cheese: "BREAKING: NASA confirms moon is made of green cheese and dairy! Secret space mission reveals government coverup!"
};

// Batch sample text
const BATCH_SAMPLE_TEXT = `India's Chandrayaan-3 Successfully Lands on the Moon near the lunar south pole.

BREAKING: NASA confirms moon is made of green cheese and dairy! Secret space mission reveals government coverup!

Solar Power Becomes One of the World's Fastest-Growing Electricity Sources as renewable investments surge.

New Smartphone Battery Technology Allows One-Year Charging without recharge, though no laboratory results exist.`;

loadBatchSampleBtn?.addEventListener('click', () => {
    if (batchText) batchText.value = BATCH_SAMPLE_TEXT;
});

// Example chips click handlers
document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const key = chip.getAttribute('data-example');
        if (SAMPLES[key] && textarea) {
            textarea.value = SAMPLES[key];
            count.textContent = `${textarea.value.length.toLocaleString()} / 30,000`;
            errorBox.hidden = true;
            resultPanel.hidden = true;
            textarea.focus();
            form.dispatchEvent(new Event('submit'));
        }
    });
});

menu?.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menu.setAttribute('aria-expanded', String(open));
});

// URL Tab Switcher & Article Extractor
tabModeText?.addEventListener('click', () => {
    tabModeText.classList.add('active');
    tabModeUrl?.classList.remove('active');
    if (urlInputBlock) urlInputBlock.style.display = 'none';
    if (textarea) textarea.focus();
});

tabModeUrl?.addEventListener('click', () => {
    tabModeUrl.classList.add('active');
    tabModeText?.classList.remove('active');
    if (urlInputBlock) urlInputBlock.style.display = 'block';
    if (articleUrlInput) articleUrlInput.focus();
});

extractUrlBtn?.addEventListener('click', async () => {
    if (!articleUrlInput) return;
    const url = articleUrlInput.value.trim();
    if (!url) {
        showUrlStatus('Please paste an article URL first.', true);
        return;
    }
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        showUrlStatus('URL must begin with http:// or https://', true);
        return;
    }

    extractUrlBtn.disabled = true;
    extractUrlBtn.textContent = 'FETCHING…';
    showUrlStatus('Scraping article text and headline…', false);

    try {
        const res = await fetch('/api/extract-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to extract article content.');

        if (textarea) {
            textarea.value = data.text;
            count.textContent = `${textarea.value.length.toLocaleString()} / 30,000`;
        }
        showUrlStatus(`✓ Extracted "${data.title.slice(0, 42)}…" (${data.word_count} words from ${data.domain})`, false);
        tabModeText?.click();
    } catch (e) {
        showUrlStatus(e.message || 'Error extracting URL', true);
    } finally {
        extractUrlBtn.disabled = false;
        extractUrlBtn.textContent = 'FETCH ARTICLE 📥';
    }
});

function showUrlStatus(msg, isError) {
    if (!urlStatus) return;
    urlStatus.style.display = 'block';
    urlStatus.textContent = msg;
    urlStatus.className = `url-status ${isError ? 'error' : ''}`;
}

// Print / Export Verification Report
exportReportBtn?.addEventListener('click', () => {
    window.print();
});

// In-Text XAI Heatmap Toggle
toggleHeatmapBtn?.addEventListener('click', () => {
    if (!xaiHeatmapWrap) return;
    if (xaiHeatmapWrap.style.display === 'block') {
        xaiHeatmapWrap.style.display = 'none';
        return;
    }

    let safeText = currentRawArticle;
    if (!currentXaiFeatures || currentXaiFeatures.length === 0) {
        xaiHeatmapWrap.textContent = safeText;
        xaiHeatmapWrap.style.display = 'block';
        return;
    }

    let html = safeText;
    currentXaiFeatures.forEach(feat => {
        const term = feat.feature.replace(/^\[char\]\s*/, '').trim();
        if (!term || term.length < 3) return;
        const regex = new RegExp(`\\b(${term})\\b`, 'gi');
        const hlClass = feat.direction === 'REAL' ? 'hl-real' : 'hl-fake';
        const tip = `Impact: ${feat.direction === 'REAL' ? '+' : ''}${feat.impact} (${feat.direction})`;
        html = html.replace(regex, `<mark class="${hlClass}" title="${tip}">$1</mark>`);
    });

    xaiHeatmapWrap.innerHTML = html;
    xaiHeatmapWrap.style.display = 'block';
});

// Update character count and disappear result when text is cleared
textarea?.addEventListener('input', () => {
    const text = textarea.value;
    count.textContent = `${text.length.toLocaleString()} / 30,000`;
    if (!text.trim()) {
        resultPanel.hidden = true;
        errorBox.hidden = true;
        if (sourcesBlock) sourcesBlock.style.display = 'none';
        if (sourcesList) sourcesList.innerHTML = '';
        if (radarBlock) radarBlock.style.display = 'none';
        if (xaiBlock) xaiBlock.style.display = 'none';
        if (xaiHeatmapWrap) xaiHeatmapWrap.style.display = 'none';
        if (comparisonBlock) comparisonBlock.style.display = 'none';
        resultPanel.classList.remove('verdict-real', 'verdict-fake', 'verdict-unverified');
    }
});

// Check backend health
fetch('/api/health')
    .then(r => r.json())
    .then(data => {
        status.textContent = data.ready ? `Engine Active · Multi-Layer XAI Platform` : 'Models initializing…';
    })
    .catch(() => {
        status.textContent = 'Server starting · ready to analyze';
    });

// Fetch analysis history on page load
loadHistory();

function loadHistory() {
    fetch('/api/history')
        .then(r => r.json())
        .then(data => {
            renderHistory(data.history || []);
        })
        .catch(() => {});
}

function renderHistory(items) {
    if (!historyList || !historyEmpty) return;
    if (!items || items.length === 0) {
        historyEmpty.style.display = 'block';
        historyList.style.display = 'none';
        historyList.innerHTML = '';
        return;
    }

    historyEmpty.style.display = 'none';
    historyList.style.display = 'flex';
    historyList.innerHTML = '';

    items.forEach((item, idx) => {
        const itemWrap = document.createElement('div');
        itemWrap.className = 'history-item';
        itemWrap.id = `history-entry-${item.id || idx}`;

        // Header
        const header = document.createElement('div');
        header.className = 'history-item-header';
        header.title = 'Click to view full analysis details';

        const left = document.createElement('div');
        left.className = 'history-item-left';

        const title = document.createElement('span');
        title.className = 'history-item-title';
        title.textContent = item.title;

        const meta = document.createElement('span');
        meta.className = 'history-item-meta';
        meta.textContent = `${item.model} · ${item.sources_count || 0} sources cited`;

        left.appendChild(title);
        left.appendChild(meta);

        const right = document.createElement('div');
        right.className = 'history-item-right';

        const badge = document.createElement('span');
        badge.className = `v-badge ${item.verdict === 'real' ? 'real' : item.verdict === 'fake' ? 'fake' : 'unverified'}`;
        badge.textContent = item.verdict === 'real' ? 'VERIFIED REAL' : item.verdict === 'fake' ? 'FLAGGED FAKE' : 'UNVERIFIED';

        const toggleIcon = document.createElement('span');
        toggleIcon.className = 'history-item-toggle';
        toggleIcon.textContent = '▼';

        right.appendChild(badge);
        right.appendChild(toggleIcon);

        header.appendChild(left);
        header.appendChild(right);

        // Body (collapsible details)
        const body = document.createElement('div');
        body.className = 'history-item-body';

        // 1. Description
        if (item.description) {
            const desc = document.createElement('p');
            desc.className = 'history-body-lead';
            desc.textContent = item.description;
            body.appendChild(desc);
        }

        // 2. Meta Pills (Model, Interpretation, Linguistic Signals)
        const pillsWrap = document.createElement('div');
        pillsWrap.className = 'history-body-meta';

        if (item.model) {
            const pillModel = document.createElement('span');
            pillModel.className = 'history-pill';
            pillModel.textContent = `Model: ${item.model}`;
            pillsWrap.appendChild(pillModel);
        }

        if (item.interpretation) {
            const pillInterp = document.createElement('span');
            pillInterp.className = 'history-pill';
            pillInterp.textContent = `Method: ${item.interpretation}`;
            pillsWrap.appendChild(pillInterp);
        }

        if (item.linguistic_signals) {
            const ls = item.linguistic_signals;
            if (ls.sensationalism_level) {
                const pillSens = document.createElement('span');
                pillSens.className = 'history-pill';
                pillSens.textContent = `Sensationalism: ${ls.sensationalism_level} (${ls.sensationalism_score || 0}%)`;
                pillsWrap.appendChild(pillSens);
            }
            if (ls.readability_label) {
                const pillRead = document.createElement('span');
                pillRead.className = 'history-pill';
                pillRead.textContent = `Style: ${ls.readability_label}`;
                pillsWrap.appendChild(pillRead);
            }
        }
        body.appendChild(pillsWrap);

        // 3. Citations & sources if present
        if (item.sources && item.sources.length > 0) {
            const srcContainer = document.createElement('div');
            srcContainer.className = 'history-sources';
            const srcTitle = document.createElement('strong');
            srcTitle.textContent = 'CITATIONS & CORROBORATING SOURCES:';
            srcContainer.appendChild(srcTitle);

            const ul = document.createElement('ul');
            item.sources.forEach(s => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = s.link || '#';
                a.target = '_blank';
                a.rel = 'noreferrer';
                a.textContent = `${s.source ? '[' + s.source + '] ' : ''}${s.title}`;
                li.appendChild(a);
                ul.appendChild(li);
            });
            srcContainer.appendChild(ul);
            body.appendChild(srcContainer);
        }

        // 4. Text snippet preview
        if (item.text) {
            const previewBox = document.createElement('div');
            previewBox.className = 'history-text-preview';
            previewBox.textContent = item.text;
            body.appendChild(previewBox);
        }

        // 5. Action: Re-analyze in full analyzer
        const actions = document.createElement('div');
        actions.className = 'history-actions';
        const reloadBtn = document.createElement('button');
        reloadBtn.type = 'button';
        reloadBtn.className = 'history-reload-btn';
        reloadBtn.textContent = 'RE-ANALYZE IN TRUTHLENS ↗';
        reloadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (textarea && item.text) {
                textarea.value = item.text;
                count.textContent = `${textarea.value.length.toLocaleString()} / 30,000`;
                textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
                textarea.focus();
                form.dispatchEvent(new Event('submit'));
            }
        });
        actions.appendChild(reloadBtn);
        body.appendChild(actions);

        // Click on header toggles expansion
        header.addEventListener('click', () => {
            const isOpen = itemWrap.classList.toggle('open');
            toggleIcon.textContent = isOpen ? '▲' : '▼';
        });

        itemWrap.appendChild(header);
        itemWrap.appendChild(body);
        historyList.appendChild(itemWrap);
    });
}

clearHistoryBtn?.addEventListener('click', async () => {
    try {
        await fetch('/api/history/clear', { method: 'POST' });
        renderHistory([]);
    } catch (e) {}
});

// SINGLE STORY ANALYZE FORM
form?.addEventListener('submit', async e => {
    e.preventDefault();
    errorBox.hidden = true;
    resultPanel.hidden = true;
    if (sourcesBlock) sourcesBlock.style.display = 'none';
    if (sourcesList) sourcesList.innerHTML = '';
    if (radarBlock) radarBlock.style.display = 'none';
    if (xaiBlock) xaiBlock.style.display = 'none';
    if (xaiHeatmapWrap) {
        xaiHeatmapWrap.style.display = 'none';
        xaiHeatmapWrap.innerHTML = '';
    }
    if (comparisonBlock) comparisonBlock.style.display = 'none';
    resultPanel.classList.remove('verdict-real', 'verdict-fake', 'verdict-unverified');

    const text = textarea.value.trim();
    if (!text) {
        showError('Please paste a headline or article text before analyzing.');
        return;
    }

    currentRawArticle = text;
    button.disabled = true;
    button.textContent = 'INVESTIGATING…';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Analysis could not be completed.');

        resultTitle.textContent = data.label;

        // Apply distinct visual themes
        if (data.verdict === 'real') {
            resultPanel.classList.add('verdict-real');
            resultTag.textContent = '✓ VERIFIED REAL NEWS';
        } else if (data.verdict === 'fake') {
            resultPanel.classList.add('verdict-fake');
            resultTag.textContent = '⚠ FAKE / MISLEADING';
        } else {
            resultPanel.classList.add('verdict-unverified');
            resultTag.textContent = 'ℹ UNVERIFIED CLAIM';
        }

        resultDesc.textContent = data.description || '';
        resultModel.textContent = data.model || 'TruthLens Hybrid Engine';
        if (resultInterp) {
            resultInterp.textContent = data.interpretation || 'Cross-Source Analysis';
        }
        resultNote.textContent = data.note || 'Always cross-reference critical claims with primary reporting.';

        // Render corroborating sources if available
        if (data.sources && data.sources.length > 0 && sourcesBlock && sourcesList) {
            sourcesList.innerHTML = '';
            data.sources.forEach(src => {
                const li = document.createElement('li');
                const sourceBadge = document.createElement('strong');
                sourceBadge.textContent = src.source ? `[${src.source}] ` : '';
                sourceBadge.style.color = '#7b736a';
                sourceBadge.style.fontFamily = 'var(--mono)';
                sourceBadge.style.fontSize = '11px';

                const a = document.createElement('a');
                a.href = src.link || '#';
                a.target = '_blank';
                a.rel = 'noreferrer';
                a.textContent = src.title || 'Source article';

                li.appendChild(sourceBadge);
                li.appendChild(a);
                sourcesList.appendChild(li);
            });
            sourcesBlock.style.display = 'block';
        } else if (sourcesBlock) {
            sourcesBlock.style.display = 'none';
        }

        // Render Linguistic Credibility & Sensationalism Radar
        if (data.linguistic_signals && radarBlock) {
            const ls = data.linguistic_signals;
            if (radarSensBadge && radarSensFill && radarSensDesc) {
                radarSensBadge.textContent = `${ls.sensationalism_level} Risk`;
                radarSensBadge.className = `radar-card-badge ${ls.sensationalism_level.toLowerCase()}`;
                radarSensFill.style.width = `${ls.sensationalism_score}%`;
                radarSensFill.className = `radar-meter-fill ${ls.sensationalism_score >= 60 ? 'alert' : ls.sensationalism_score >= 28 ? 'warn' : ''}`;
                radarSensDesc.textContent = `Score: ${ls.sensationalism_score}% · ${ls.sensational_words && ls.sensational_words.length ? 'Trigger terms: ' + ls.sensational_words.join(', ') : 'Calm journalistic tone'}`;
            }
            if (radarToneBadge && radarToneDesc && radarTonePronouns) {
                radarToneBadge.textContent = ls.tone_label.includes('Subjective') ? 'Opinionated' : 'Neutral';
                radarToneBadge.className = `radar-card-badge ${ls.tone_label.includes('Subjective') ? 'moderate' : 'neutral'}`;
                radarToneDesc.textContent = ls.tone_label;
                radarTonePronouns.textContent = `Word count: ${ls.word_count} words`;
            }
            if (radarCapsBadge && radarUrgencyDetail) {
                radarCapsBadge.textContent = (ls.all_caps_count > 0 || ls.exclamation_count > 0) ? 'Attention Signs' : 'Normal';
                radarCapsBadge.className = `radar-card-badge ${(ls.all_caps_count > 2 || ls.exclamation_count > 2) ? 'high' : 'neutral'}`;
                radarUrgencyDetail.textContent = `${ls.all_caps_count} uppercase words · ${ls.exclamation_count} exclamation marks`;
            }
            if (radarReadBadge && radarReadLabel && radarReadDesc) {
                radarReadBadge.textContent = ls.readability_label.includes('Investigative') ? 'Advanced' : 'Standard';
                radarReadLabel.textContent = ls.readability_label;
                radarReadDesc.textContent = `${ls.readability_score}`;
            }
            radarBlock.style.display = 'block';
        } else if (radarBlock) {
            radarBlock.style.display = 'none';
        }

        // Render Explainable AI (XAI) feature attributions
        currentXaiFeatures = (data.explainable_ai && data.explainable_ai.features) || [];
        if (currentXaiFeatures.length > 0 && xaiBlock && xaiBars) {
            xaiBars.innerHTML = '';
            const maxAbsImpact = Math.max(...currentXaiFeatures.map(f => Math.abs(f.impact)), 0.01);

            currentXaiFeatures.forEach(f => {
                const row = document.createElement('div');
                row.className = 'xai-bar-row';

                const name = document.createElement('span');
                name.className = 'xai-feat-name';
                name.textContent = f.feature;
                name.title = f.feature;

                const track = document.createElement('div');
                track.className = 'xai-bar-track';

                const fill = document.createElement('div');
                const pct = Math.min(Math.round((Math.abs(f.impact) / maxAbsImpact) * 100), 100);
                fill.className = `xai-bar-fill ${f.direction === 'REAL' ? 'real' : 'fake'}`;
                fill.style.width = `${Math.max(pct, 8)}%`;
                track.appendChild(fill);

                const badge = document.createElement('span');
                badge.className = `xai-badge ${f.direction === 'REAL' ? 'real' : 'fake'}`;
                badge.textContent = `${f.direction === 'REAL' ? '+REAL' : '-FAKE'} (${f.impact > 0 ? '+' : ''}${f.impact})`;

                row.appendChild(name);
                row.appendChild(track);
                row.appendChild(badge);
                xaiBars.appendChild(row);
            });
            xaiBlock.style.display = 'block';
        } else if (xaiBlock) {
            xaiBlock.style.display = 'none';
        }

        // Render Multi-Classifier Comparison Benchmark
        if (data.explainable_ai && data.explainable_ai.comparison && comparisonBlock) {
            const comp = data.explainable_ai.comparison;
            if (comp.svm && compSvmPred) {
                compSvmPred.textContent = comp.svm.prediction;
                compSvmPred.className = `c-pred ${comp.svm.prediction.toLowerCase()}`;
                compSvmConf.textContent = `Margin score: ${comp.svm.margin_score}`;
            }
            if (comp.logistic_regression && compLrPred) {
                compLrPred.textContent = comp.logistic_regression.prediction;
                compLrPred.className = `c-pred ${comp.logistic_regression.prediction.toLowerCase()}`;
                compLrConf.textContent = `Confidence: ${comp.logistic_regression.confidence}%`;
            }
            if (comp.naive_bayes && compNbPred) {
                compNbPred.textContent = comp.naive_bayes.prediction;
                compNbPred.className = `c-pred ${comp.naive_bayes.prediction.toLowerCase()}`;
                compNbConf.textContent = `Confidence: ${comp.naive_bayes.confidence}%`;
            }
            comparisonBlock.style.display = 'block';
        } else if (comparisonBlock) {
            comparisonBlock.style.display = 'none';
        }

        resultPanel.hidden = false;
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Refresh history list
        loadHistory();
    } catch (err) {
        showError(err.message || 'Check that the Flask server is running.');
    } finally {
        button.disabled = false;
        button.textContent = 'ANALYZE STORY ↗';
    }
});

// BATCH TESTING FORM HANDLER
batchSubmitBtn?.addEventListener('click', async () => {
    if (!batchText) return;
    const raw = batchText.value.trim();
    if (!raw) {
        alert('Please paste some text into the batch testing area first.');
        return;
    }

    // Split by empty lines or double newlines into distinct articles
    const rawArticles = raw.split(/\n\s*\n/).map(a => a.trim()).filter(a => a.length > 5);
    if (rawArticles.length === 0) {
        alert('Could not find distinct articles. Separate each article with an empty line.');
        return;
    }

    batchSubmitBtn.disabled = true;
    batchSubmitBtn.textContent = 'RUNNING BATCH ANALYSIS…';

    try {
        const resp = await fetch('/api/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ articles: rawArticles })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Batch analysis failed.');

        if (batchTotal) batchTotal.textContent = data.summary.total;
        if (batchReal) batchReal.textContent = data.summary.real;
        if (batchFake) batchFake.textContent = data.summary.fake;
        if (batchUnverified) batchUnverified.textContent = data.summary.unverified;
        if (batchScore) batchScore.textContent = `${data.summary.credibility_score}%`;

        if (batchTableBody) {
            batchTableBody.innerHTML = '';
            data.results.forEach(r => {
                const tr = document.createElement('tr');

                const tdIdx = document.createElement('td');
                tdIdx.textContent = r.index;
                tdIdx.style.fontFamily = 'var(--mono)';
                tdIdx.style.color = '#7b736a';

                const tdText = document.createElement('td');
                tdText.textContent = r.snippet + '…';

                const tdVerdict = document.createElement('td');
                const badge = document.createElement('span');
                badge.className = `v-badge ${r.verdict === 'real' ? 'real' : r.verdict === 'fake' ? 'fake' : 'unverified'}`;
                badge.textContent = r.verdict === 'real' ? 'VERIFIED REAL' : r.verdict === 'fake' ? 'FLAGGED FAKE' : 'UNVERIFIED';
                tdVerdict.appendChild(badge);

                const tdModel = document.createElement('td');
                tdModel.textContent = r.model;
                tdModel.style.fontSize = '11px';
                tdModel.style.color = '#554e46';

                tr.appendChild(tdIdx);
                tr.appendChild(tdText);
                tr.appendChild(tdVerdict);
                tr.appendChild(tdModel);
                batchTableBody.appendChild(tr);
            });
        }

        if (batchResultsWrap) {
            batchResultsWrap.style.display = 'block';
            batchResultsWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        // Refresh history
        loadHistory();
    } catch (err) {
        alert(err.message || 'Error processing batch.');
    } finally {
        batchSubmitBtn.disabled = false;
        batchSubmitBtn.textContent = 'RUN BATCH ANALYSIS ↗';
    }
});

function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
}

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    revealItems.forEach(el => observer.observe(el));
} else {
    revealItems.forEach(el => el.classList.add('is-visible'));
}
