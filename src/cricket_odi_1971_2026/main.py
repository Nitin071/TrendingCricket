import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://assets-icc.sportz.io/cricket/v1/ranking"

PARAMS_BASE = {
    "client_id": "tPZJbRgIub3Vua93/DWtyQ==",
    "comp_type": "odi",
    "feed_format": "json",
    "lang": "en",
    "type": "bat",
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.icc-cricket.com",
    "Referer": "https://www.icc-cricket.com/",
}

# Parallel HTTP workers (tune down if the API rate-limits)
MAX_WORKERS = 16
REQUEST_TIMEOUT = 60

start_year = 1971
end_year = 2026


def build_jobs():
    jobs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == 2026 and month > 3:
                break
            jobs.append((year, month))
    return jobs


def fetch_month(job):
    year, month = job
    date_str = f"{year}{month:02d}01"
    params = {**PARAMS_BASE, "date": date_str}
    rows = []
    try:
        response = requests.get(
            BASE_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT
        )
        json_data = response.json()
        api_data = json_data.get("data")
        if not api_data or response.status_code != 200:
            return year, month, date_str, rows, f"Skip (no data): {date_str} status={response.status_code}"

        bat = api_data.get("bat-rank") or {}
        rankings = (bat.get("rank") or [])[:15]
        for player in rankings:
            rows.append({
                "Date": f"{year}-{month:02d}",
                "Player": player.get("Player-name"),
                "Country": player.get("Country_name"),
                "Points": player.get("Points"),
            })
        return year, month, date_str, rows, f"Done: {date_str}"
    except Exception as e:
        return year, month, date_str, rows, f"Error at {date_str}: {e}"


def main():
    jobs = build_jobs()
    data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_month, j): j for j in jobs}
        for fut in as_completed(futures):
            year, month, date_str, rows, msg = fut.result()
            print(msg)
            data.extend(rows)

    data.sort(key=lambda r: (r["Date"], r["Player"] or ""))
    df = pd.DataFrame(data)
    df.to_csv("odi_batting_rankings.csv", index=False)
    print("✅ Data saved!")


if __name__ == "__main__":
    main()
