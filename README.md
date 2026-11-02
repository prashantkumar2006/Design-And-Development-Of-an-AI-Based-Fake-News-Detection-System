# News Fact Checker

This app checks a headline or short claim against live web sources using the OpenAI Responses API web search tool. The older TF-IDF classifier files remain available for coursework, but the app does not use them to decide whether a claim is true or false.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Paste an OpenAI API key into the password field in the app sidebar. The key is kept in the current Streamlit session and is not written to a project file. You can instead set `OPENAI_API_KEY` in the environment before starting Streamlit.

An OpenAI API key and API usage access are required; a ChatGPT subscription does not provide API access. API requests use live web search, so each new claim may incur API usage charges. The app sends the claim to OpenAI for search and analysis. Do not enter private or confidential information.

## Verdict rules

- **Supported**: retrieved sources directly support the central claim.
- **Contradicted**: retrieved sources directly conflict with it.
- **Mixed**: reliable evidence supports and contradicts material parts, or the claim needs important qualification.
- **Unverified**: the evidence is missing, weak, stale, conflicting, or cannot be tied to retrieved sources.

Decisive labels are downgraded to **Unverified** unless the supporting or contradicting evidence ties to a primary source or at least two distinct retrieved publisher domains. This conservative check reduces unsupported verdicts; it cannot guarantee that every source or conclusion is correct. Open the cited sources for important claims.
