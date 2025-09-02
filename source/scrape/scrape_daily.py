import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from zipfile import ZipFile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Chrome
from tqdm import tqdm
import shutil

# --- SETTINGS ---
START_DATE = datetime(1994, 1, 25)
END_DATE = datetime(2025, 7, 2)
RESCRAPE_EARLY = True  # <-- Set to True to force rescrape

DOWNLOAD_DIR = "/Users/AndrewKao/Downloads"    
STORAGE_DIR = "/Users/AndrewKao/Documents/Grad/staffers/datastore/scrape/cr-daily"       
CSV_FILE = STORAGE_DIR + "/link_list.csv"

# --- SELENIUM SETUP ---
chrome_options = Options()
chrome_options.add_experimental_option('prefs', {
    "download.default_directory": DOWNLOAD_DIR,  # keep separate for clean directories
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
})
driver = Chrome('datastore/chromedriver-mac-arm64/chromedriver', options=chrome_options)

# --- GENERATE DATE RANGE ---
def generate_date_links(start_date, end_date):
    days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(days)]
    urls = [
        f"https://www.govinfo.gov/content/pkg/CREC-{dt.strftime('%Y-%m-%d')}.zip"
        for dt in dates
    ]
    return dates, urls

# --- INIT CSV ---
def init_csv(dates, urls):
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "url": urls,
            "applicable": None,
            "completed": False,
            "error": ""
        })
        df.to_csv(CSV_FILE, index=False)

def load_progress():
    return pd.read_csv(CSV_FILE)

def save_progress(df):
    df.to_csv(CSV_FILE, index=False)

# --- DOWNLOAD UTILS ---
def is_downloaded(filename):
    return os.path.exists(os.path.join(DOWNLOAD_DIR, filename))

def wait_for_download(filename, timeout=900):
    target = os.path.join(DOWNLOAD_DIR, filename)
    for _ in range(timeout):
        if os.path.exists(target):
            # File might still be writing, wait a bit
            time.sleep(2)
            return True
        time.sleep(1)
    return False

def move_and_unzip(filename):
    import shutil
    src = os.path.join(DOWNLOAD_DIR, filename)
    dest_dir = os.path.join(STORAGE_DIR, filename.replace('.zip', ''))
    os.makedirs(STORAGE_DIR, exist_ok=True)
    try:
        # Move zip
        dest_zip = os.path.join(STORAGE_DIR, filename)
        os.rename(src, dest_zip)
        return True
    except Exception as e:
        return str(e)


def check_url_exists(url, retries=3, delay=10):
    headers = {'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36')}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            if '/error' in r.url:
                return False
            # Accept if ZIP file and not tiny/empty
            if r.status_code == 200 and 'zip' in r.headers.get('Content-Type', '').lower() and len(r.content) > 1000:
                return True
            if r.status_code == 404:
                return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"URL check failed for {url}: {e}")
    return False

    
# --- MAIN LOOP ---
def main():
    dates, urls = generate_date_links(START_DATE, END_DATE)
    init_csv(dates, urls)
    df = load_progress()
    cutoff_date = datetime(2025, 7, 31)

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        date_str, url = row['date'], row['url']
        zip_name = url.split("/")[-1]
        row_date = datetime.strptime(date_str, "%Y-%m-%d")

        # Logic: Force re-scrape on or before cutoff date if flag is set
        rescrape_this_row = False
        if RESCRAPE_EARLY and row_date <= cutoff_date:
            rescrape_this_row = True

        # Skipping logic
        if not rescrape_this_row:
            if pd.notna(row['applicable']) and row['applicable'] is False:
                continue
            if row['completed']:
                continue

        # Step 1: Check if applicable
        if not check_url_exists(url):
            df.at[idx, "applicable"] = False
            save_progress(df)
            print(f"{date_str}: Not applicable (no record)")
            continue
        else:
            df.at[idx, "applicable"] = True

        # Step 2: Download ZIP
        try:
            # Remove pre-existing partial downloads if any
            tmp_path = os.path.join(DOWNLOAD_DIR, zip_name)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"{date_str}: Downloading {zip_name} ...")
            driver.get(url)
            # Wait for download to complete
            if not wait_for_download(zip_name, timeout=600):  # up to 10 min
                raise Exception("Timeout waiting for download")
        except Exception as e:
            df.at[idx, "error"] = f"Download error: {str(e)}"
            save_progress(df)
            print(f"{date_str}: ERROR {str(e)}")
            continue

        # Step 3: Move and Unzip
        try:
            result = move_and_unzip(zip_name)
            df.at[idx, "completed"] = True
            if result is not True:
                df.at[idx, "error"] = "Unzip/structure error: " + str(result)
                print(f"{date_str}: Completed with ERR {str(result)}")
            else:
                print(f"{date_str}: Completed")
        except Exception as e:
            df.at[idx, "error"] = f"Move/unzip error: {str(e)}"
            print(f"{date_str}: ERROR {str(e)}")
        save_progress(df)

    driver.quit()

if __name__ == "__main__":
    main()
