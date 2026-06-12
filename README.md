# AI Paper Reader

A Streamlit app for reading academic PDFs with LLM assistance. Upload one or more research papers, generate structured reading cards, ask paper-specific questions, compare saved cards, and export your notes.

## Tech Stack

- Python
- Streamlit
- LangChain and ChatOpenAI
- OpenAI-compatible API providers
- PyMuPDF for PDF text extraction
- Pandas and scikit-learn
- fpdf2 for lightweight PDF export

## Project Structure

```text
main.py          # Streamlit app and UI flow
utils.py         # PDF parsing, LLM calls, exports, and local persistence helpers
prompts.py       # LangChain prompt templates
requirements.txt # Python dependencies
data/.gitkeep    # Keeps the local data folder in the repository
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and add your credentials:

```bash
copy .env.example .env
```

4. Fill in `.env`:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
MODEL_NAME=
```

`.env` is optional for local development. The Streamlit sidebar starts with empty API fields unless you have explicitly saved settings from the app.

Enter API settings directly in the Streamlit sidebar. The API key input is a password field. Saved settings are stored locally in `data/api_settings.json`, which is ignored by Git and should not be committed.

5. Run:

```bash
streamlit run main.py
```

The app will still open without API settings. PDF parsing works locally, while LLM features show a warning until API Key, API URL, and Model Name are entered.

## Features

- Upload multiple PDF research papers.
- Extract selectable PDF text page by page using PyMuPDF.
- Generate one reading card per processed paper.
- Save reading cards locally in `data/reading_cards.json`.
- Ask questions grounded in a selected paper's text.
- Compare saved reading cards in a paper summary table.
- Delete processed papers from the active workspace.
- Export the reading matrix, Q&A history, Markdown report, and PDF report.

## Local Files and Privacy

The repository is configured to exclude local credentials and generated files, including `.env`, saved API settings, reading-card JSON records, uploaded PDFs, cache folders, and exported reports. Keep API keys and paper files local.

## Example Questions

- What is the main research question?
- What method and data does the paper use?
- What are the key findings?
- What limitations do the authors mention?
- Why is this paper relevant to my topic?

## Error Handling

The MVP handles:

- Missing API settings
- No PDF uploaded
- Scanned or empty PDFs
- Failed PDF parsing
- Failed LLM calls
- Q&A before upload
- Comparison with fewer than two processed papers

## Limitations

- Scanned PDFs need OCR before the app can read them.
- Long papers are truncated before structured LLM calls to keep prompts manageable.
- The comparison table uses saved reading cards, so each paper should have a card before comparison.
- The first PDF export version is intended for English text.

## Future Improvements

- Add OCR for scanned papers.
- Support citation extraction and bibliography parsing.
- Add multi-paper synthesis prompts.
- Add confidence scores and source coverage checks.
