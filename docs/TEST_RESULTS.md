# Test results

## Environment

- Baseline revision: `c6ce11c`
- Python: 3.14.6
- Node.js: 24.19.0

## Results

| Check | Status | Notes |
| --- | --- | --- |
| Automated core tests | Passed | `4 passed in 0.02s` using `python -m pytest tests -q`. |
| Frontend production build | Blocked by local environment | `npm.cmd install` could not write to the Windows npm cache (`EPERM`); Vite was therefore unavailable. This is not an application build result. |
| API health endpoint | Pending | Requires FastAPI dependencies. |
| LLM-backed text route | Not run | An authorized `GROQ_API_KEY` was not provided. |
| SQL route | Passed | Created the sample database and verified Apple, Microsoft, NVIDIA prices and Apple 2024 revenue. |
| Vision route | Pending | Requires runtime dependencies. |

The automated suite intentionally covers API-key-free logic. The manual scenarios in `TESTING.md` cover integrations requiring external credentials or running services.
