# Testing guide

## Automated checks

```powershell
python -m pytest tests -q
```

These deterministic checks do not call Groq. They validate chunk overlap, search ranking, and relevance selection.

## Manual end-to-end checks

After configuring `GROQ_API_KEY`, creating the database, and starting the API and frontend, verify:

| Scenario | Input | Expected result |
| --- | --- | --- |
| Document question | `What is the Week 2 development plan?` | Text route, sources, grounded answer |
| Missing fact | `What does the document say about revenue?` | Retry and safe not-found response if unsupported |
| SQL query | `What is Apple's stock price?` | SQL route and sample-data response |
| Vision query | `What does this chart show?` | Vision route and extracted image URL |
| Health | `GET /health` | `{"status":"healthy"}` |
