import os 
import time 
import json 
import requests 
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BRIGHTDATA_API_KEY")
DATASET_ID = "gd_lpfll7v5hcqtkxl6l"
AUTH = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# roles we want to pull job desccriptions for 
roles = [
    {
        "keyword": "data engineer", 
        "location": "Canada",
        "country": "CA",
        "remote": ""
    },

    {
        "keyword": "data scientist", 
        "location": "Canada",
        "country": "CA",
        "remote": ""  
    },

    {
        "keyword": "machine learning engineer", 
        "location": "Canada",
        "country": "CA",
        "remote": ""  
    },

    {
        "keyword": "analytics engineer", 
        "location": "Canada",
        "country": "CA",
        "remote": ""  
    }, 

    {
        "keyword": "ai engineer", 
        "location": "Canada",
        "country": "CA",
        "remote": ""  
    }
]
if not API_KEY: 
    raise ValueError("Missing BRIGHTDATA_API_KEY in environment variables")

inputs = []

# for each role, add the parameters for the roles you want to search for
for role in roles:
    inputs.append({
        "location": role['location'],
        "keyword": role["keyword"],
        "country": role["country"],
        "time_range": "",
        "job_type": "Full-time",
        "experience_level": "",
        "remote": role["remote"],
        "company": "",
        "location_radius": "",
        "selective_search": False
    })

# URL to trigger the searching for the above roles 
trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={DATASET_ID}&include_errors=true&type=discover_new&discover_by=keyword&limit_per_input=1000&limit_multiple_results=10000"
request = requests.post(trigger_url, headers = AUTH, data = json.dumps(inputs))
request.raise_for_status()
resp = request.json()
snapshot_id = resp.get("snapshot_id")

if not snapshot_id:
    raise RuntimeError(f"Could not find snapshot_id in response: {resp}")
print("Snapshot id:", snapshot_id)

# 2) poll progress
progress_url = f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
sleep = 2
while True:
    r = requests.get(progress_url, headers=AUTH, timeout=30)
    if r.status_code in (202, 204) or not r.text.strip():
        print("status: pending")
    elif r.ok:
        try:
            pj = r.json()
        except requests.JSONDecodeError:
            print("status: waiting")
            time.sleep(sleep)
            sleep = min(15, sleep * 1.5)
            continue
        status = pj.get("status", "unknown")
        print("status:", status)
        if status in {"ready", "done", "finished", "success"}:
            break
    else:
        print(f"status: HTTP {r.status_code}, body: {r.text[:200]}")
    time.sleep(sleep)
    sleep = min(15, sleep * 1.5)

# 3) download snapshot as JSON
download_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"
d = requests.get(download_url, headers=AUTH, timeout=120, allow_redirects=True)
d.raise_for_status()

def save_pretty(resp, path):
    text = resp.text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError: 
        # handle NDJSON by convering it to a JSON array 
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(obj, f, ensure_ascii = False, indent = 2)

save_pretty(d, "linkedin_jobs.json")
print("Saved linkedin_jobs.json")