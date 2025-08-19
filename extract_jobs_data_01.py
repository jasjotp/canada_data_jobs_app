import os 
import time 
import json 
import requests 
from dotenv import load_dotenv
from datetime import datetime, timezone 

load_dotenv()

API_KEY = os.getenv("BRIGHTDATA_API_KEY")
DATASET_ID = "gd_lpfll7v5hcqtkxl6l"
AUTH = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# set a recency window in the dataset 
TIME_RANGE = "Past 24 hours"

now = datetime.now(timezone.utc).timestamp()

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
        "time_range": TIME_RANGE,
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
d = requests.get(download_url, headers = AUTH, timeout = 120, allow_redirects = True)
d.raise_for_status()

# function to change format of job_posted_date to Unix seconds 
def parse_posted_ts(record) -> float: 
    """
    parse job_posted_date like '2025-07-20T03:13:17.319Z' to Unix seconds.
    returns 0.0 if missing or unparseable.
    """
    job_posted_date = record.get("job_posted_date")
    if not job_posted_date:
        return 0.0 
    try: 
        # convert the trailing Z as UTC 
        return datetime.fromisoformat(str(job_posted_date).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

# function to save the response as a formatted JSON object
def save_pretty(resp):
    text = resp.text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError: 
        # handle NDJSON by convering it to a JSON array 
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    return obj

def deduplicate_and_filter_24h(records):
    """
    Keeps only records posted in the last 24 hours. 
    If there is a duplicate job posting ID, we keep the most recent posting if duplicate job ids appear.
    """
    cutoff = now - 24 * 3600 # cutoff time that the records job posing date should be greater than (within last 24h) 
    
    # dictionary with job_posting_id: record data
    by_id = {}

    for record in records:
        job_id = record.get("job_posting_id")
        if not job_id: # if there is no job id, skip that record
            continue 
        posted_ts = parse_posted_ts(record)

        # if the job was posted more than 24h ago, skip that record 
        if posted_ts < cutoff:
            continue 
        
        # keep the most recent version of a duplicate id 
        curr = by_id.get(job_id)
        if not curr or posted_ts > parse_posted_ts(curr):
            record["posted_ts"] = posted_ts
            by_id[job_id] = record

    return list(by_id.values())

# function that keeps only records 
raw_list = save_pretty(d)
print(f"Downloaded {len(raw_list)} records")

# filter and deduplicate the records 
clean_list = deduplicate_and_filter_24h(raw_list)
print(f"Kept {len(clean_list)} records from the last 24 hours after dedupe")

# sort the newest jobs first 
clean_list.sort(key = lambda record: record.get("posted_ts", 0), reverse = True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_path = f"linkedin_jobs_last24h_{timestamp}.json"

with open(output_path, "w", encoding = "utf-8") as f:
    json.dump(clean_list, f, ensure_ascii = False, indent = 2)

print(f"Saved {output_path}")
