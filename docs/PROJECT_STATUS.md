# Project status

## Week 4 handoff status

| Area | Status | Evidence |
| --- | --- | --- |
| README and project documentation | Complete | Root README plus architecture and testing documents |
| Automated regression checks | Complete | `python -m pytest tests -q` — 4 passing tests |
| SQL sample workflow | Complete | Sample database created and stock/revenue queries verified |
| Repository organization | Complete | Duplicate extraction project removed; configuration template and ignore rules added |
| Frontend production build | Environment-blocked | Windows npm-cache permissions prevented dependency installation in the validation environment |
| Groq-backed text end-to-end flow | Requires configuration | A valid `GROQ_API_KEY` must be placed in `.env` before testing LLM-backed routes |

## Reviewer instructions

1. Configure `GROQ_API_KEY` in `.env` from `.env.example`.
2. Run `python create_database.py`.
3. Run `python -m pytest tests -q`.
4. Start the API with `uvicorn main:app --reload`.
5. Install and build the frontend from `frontend/` with `npm install` and `npm run build`.

The Week 4 documentation, testing, and repository-organization handoff is credited to Kundan in commit `11cef18` and its follow-up documentation commit.
