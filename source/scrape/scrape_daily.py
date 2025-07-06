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

# --- SETTINGS ---
START_DATE = datetime(1994, 1, 25)
END_DATE = datetime(2025, 7, 2)
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

def wait_for_download(filename, timeout=300):
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
        # Unzip
        with ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        os.remove(dest_zip)
        # Handle nested directory
        subfolder = os.path.join(dest_dir, os.path.basename(dest_dir))
        if os.path.isdir(subfolder):
            # Move all contents up
            for item in os.listdir(subfolder):
                shutil.move(os.path.join(subfolder, item), dest_dir)
            os.rmdir(subfolder)
        # Check structure
        html_dir = os.path.join(dest_dir, 'html')
        if not os.path.isdir(html_dir):
            return False
        return True
    except Exception as e:
        return str(e)


def check_url_exists(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            r = requests.head(url, allow_redirects=True, timeout=10)
            if (r.status_code == 200) and ('/error' not in r.url):
                return True
            # Some servers may misbehave with HEAD, so try GET
            r = requests.get(url, stream=True, allow_redirects=True, timeout=10)
            if (r.status_code == 200) and ('/error' not in r.url):
                return True
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

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(row['applicable']) and row['applicable'] is False:
            continue
        if row['completed']:
            continue

        date_str, url = row['date'], row['url']
        zip_name = url.split("/")[-1]

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
