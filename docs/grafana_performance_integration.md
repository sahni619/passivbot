# Grafana integration for risk-management metrics

This guide explains how to expose the risk-management performance metrics to Grafana dashboards. It walks through preparing the realtime tracker, running the FastAPI service, authenticating, wiring up a Grafana data source, and building panels for the provided equity analytics.

## 1. Prerequisites

1. **Risk-management dependencies** – install the dashboard extras so the FastAPI server and realtime fetcher are available:
   ```bash
   pip install -r requirements.txt
   ```
2. **Grafana server** – running Grafana v9+ (local or remote).
3. **JSON API data source plugin** – install [`grafana-json-datasource`](https://grafana.com/grafana/plugins/marcusolsson-json-datasource/) on your Grafana instance. The performance endpoints return JSON documents, and this plugin lets Grafana poll arbitrary HTTP APIs.
4. **API keys file** – populate `api-keys.json` with the credentials for every account you monitor (see `api-keys.json.example`).

## 2. Generate daily balance snapshots

The realtime fetcher persists portfolio and per-account balances once the target cut-off (4 pm New York time) has passed. Verify that:

1. Your realtime configuration sets the accounts you want to monitor and points to a writable `reports_dir` (defaults to `<config directory>/reports`).
2. The fetcher is running continuously so it can capture the first post–4 pm snapshot.

### 2.1 Configure realtime access

Copy the template and edit it with your account names plus `api_key_id` references:
```bash
cp risk_management/realtime_config.example.json risk_management/realtime_config.local.json
```
Update the new file with:

- `accounts` entries referencing keys in `api-keys.json`.
- `reports_dir` pointing at the directory Grafana will read from (for example `"../risk_reports"`).
- `auth.users` containing the bcrypt-hashed password for the Grafana service account you will use later.

Refer to [risk_management/README.md](../risk_management/README.md) for the full schema description.

### 2.2 Run the realtime fetcher

Launch either the terminal dashboard or the web server; both spin up the realtime fetcher which records balances:
```bash
python -m risk_management.dashboard \
  --realtime-config risk_management/realtime_config.local.json \
  --interval 60 --iterations 0
```
Let the process run through the 4 pm ET window. The tracker writes `daily_balances.json` beneath `reports_dir`, storing portfolio and per-account balance history used by the metrics API.【F:risk_management/performance.py†L22-L133】【F:risk_management/services/performance_repository.py†L20-L104】

You can inspect the latest snapshot manually:
```bash
jq '.' risk_reports/daily_balances.json
```
Adjust the path to match your configured directory.

## 3. Start the FastAPI dashboard service

Run the authenticated web server so Grafana can poll its JSON endpoints:
```bash
python -m risk_management.web_server \
  --config risk_management/realtime_config.local.json \
  --host 0.0.0.0 --port 8000
```
The server exposes `/api/performance/metrics`, `/api/performance/portfolio`, and `/api/performance/accounts/<name>` for Grafana consumption. All routes require a logged-in session.【F:risk_management/web.py†L274-L608】

## 4. Create a Grafana service session

The API uses cookie-based sessions. Create a dedicated user in `auth.users` (for example `"grafana"`) with a long, random password hash. Use `risk_management/scripts/hash_password.py` to generate the bcrypt hash.

Log in once to obtain a session cookie that Grafana can reuse:
```bash
curl -k -c grafana.session \
  -X POST https://risk-dashboard.example.com/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data 'username=grafana&password=<plain-text-password>'
```
Copy the `risk_dashboard_session=...` value from `grafana.session`. The cookie stays valid for 2 weeks by default because it is managed by Starlette’s `SessionMiddleware` using your `auth.secret_key`.【F:risk_management/web.py†L392-L438】 Repeat the login whenever you rotate the password or the cookie expires.

> **Tip:** If you operate behind a reverse proxy, limit the service user to Grafana’s IPs and prefer HTTPS (`auth.https_only=true`) so cookies are never transmitted in plain text.

## 5. Configure Grafana’s JSON API data source

1. In Grafana, navigate to **Configuration → Data sources → Add data source → JSON API**.
2. Set **URL** to your dashboard origin (for example `https://risk-dashboard.example.com`).
3. Under **Headers**, add a key named `Cookie` with value `risk_dashboard_session=<copied-session-token>`.
4. Optional: set **Allowed hosts** to the same origin to avoid SSRF warnings.
5. Save & test. Grafana should report `OK` if the cookie is valid.

## 6. Build panels for the performance metrics

Create a new dashboard and add panels using the JSON API data source. The following requests map directly to the helper functions in `risk_management/performance_metrics.py`:

### 6.1 Portfolio analytics

- **Method:** GET
- **URL:** `/api/performance/metrics?start=2024-01-01&end=2024-12-31`
- **Transformations:**
  - Use Grafana’s **JSONPath** extraction to pull `$.portfolio.equity_curve[*]` for time series panels.
  - Extract `$.portfolio.statistics.total_return_pct`, `$.portfolio.max_drawdown.percentage`, and `$.portfolio.sharpe_ratio` for single-stat panels.

The endpoint returns the equity curve, daily returns, Sharpe ratio, drawdown statistics, and summary metadata computed by `build_performance_metrics()`.【F:risk_management/performance_metrics.py†L12-L204】

### 6.2 Per-account analytics

- **Method:** GET
- **URL:** `/api/performance/metrics?account=<account-name>` (optional `start`/`end` filters)
- **Transformations:**
  - Iterate over `$.accounts.*.equity_curve[*]` for individual account panels.
  - Use Grafana’s **Reduce** or **Stat** transformations to highlight each account’s `max_drawdown.amount` or `statistics.total_return_pct`.

### 6.3 Raw balance history

If you prefer to chart the raw daily balances yourself, request `/api/performance/accounts/<account-name>?start=...&end=...` or `/api/performance/portfolio?...` to obtain the underlying series without additional analytics.【F:risk_management/services/performance_repository.py†L28-L60】

## 7. Validate data flow

After the panels are set up:

1. Trigger a manual fetch to confirm the repository is updating:
   ```bash
   curl -b grafana.session 'https://risk-dashboard.example.com/api/performance/metrics?account=<account-name>' | jq '.'
   ```
2. Refresh the Grafana dashboard and verify that the plots and single-stat cards show the latest equity curve and ratios.
3. Schedule Grafana’s data source refresh interval (for example every 15 minutes) to balance timeliness against API load.

## 8. Troubleshooting

- **No data after 4 pm:** Ensure the realtime fetcher ran past the 4 pm ET threshold. The tracker only records a new point once the localised timestamp exceeds the configured cut-off.【F:risk_management/performance.py†L38-L118】
- **401 errors in Grafana:** Recreate the session cookie or confirm the dashboard URL matches the cookie’s domain and HTTPS settings.
- **Missing accounts:** Verify `reports_dir/daily_balances.json` contains an entry under `accounts` with the expected name; otherwise, the metrics endpoint skips that account.【F:risk_management/services/performance_repository.py†L28-L75】

Following these steps provides a repeatable path from realtime exchange polling to Grafana visualisations without manual exports.
