import os
import pandas as pd
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
import pdb
import time
import re
import psutil


### downloads PDFs of the Congressional Record (bound version)
def Main():
    raw_dir = "datastore/scrape/cr-bound"

    # note a single run will miss a small number of links
    # run 3 types to make it likely that all links are scraped
    link_list = GetLinks(raw_dir, redo = 0)
    link_list = GetLinks(raw_dir, redo = 1)
    link_list = GetLinks(raw_dir, redo = 1)

    DownloadPDFs(link_list)

# gets a list of links to Congressional Record bound directories to download
# infolder: path of folder containing list of links
# redo: binary, 1 if re-scraping link list (not first time)
def GetLinks(infolder, redo = 0):

    if (redo == 1) or (not os.path.isfile('datastore/scrape/cr-bound/link_list.csv')):
        # set up webdriver 
        chrome_options =ChromeOptions()
        # chrome_options.add_argument('--headless')
        chrome_options.add_experimental_option('w3c', True)
        driver = Chrome('datastore/chromedriver_mac64/chromedriver', options=chrome_options)

        # get web page
        driver.get("https://www.govinfo.gov/app/collection/crecb")
        time.sleep(20) # seconds


        # Loop through all the panel headings and click them to expand the panels
        # source of some missed links when panels move
        panel_headings = driver.find_elements(by=By.CLASS_NAME, value="panel-heading")
        for panel_heading in panel_headings:
            time.sleep(5) # seconds
            panel_heading.click()

        # get href of buttons
        buttons = driver.find_elements(by=By.CLASS_NAME, value="btn")
        links = [button.get_attribute('href') for button in buttons]
        # only want pdf
        link_list = [l for l in links if l is not None and ".pdf" in l]

        df_link = pd.DataFrame({'link': link_list})

        if redo == 0:
            # write to disk
            df_link.to_csv('datastore/scrape/cr-bound/link_list.csv') 
        elif redo == 1:
            df_link_old = pd.read_csv('datastore/scrape/cr-bound/link_list.csv')
            # merge
            df_link = pd.merge(df_link,df_link_old,on='link',how='outer')
            # replace column 'download' with 0 when missing
            df_link = df_link.fillna(0)
            # write to disk
            df_link.to_csv('datastore/scrape/cr-bound/link_list.csv') 


    return 'datastore/scrape/cr-bound/link_list.csv'

# downloads a file via URL with progress bar
# source:
## https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
def download(url, filename):
    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm
    
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path

# downloads list of links
def DownloadPDFs(link_list):

    # get data
    df_link = pd.read_csv(link_list)
    df_link.drop_duplicates(subset = ['link'], inplace = True)

    if 'downloaded' not in df_link.columns:
        df_link['downloaded'] = 0

    # between last backslash and pdf
    pattern = r"/([^/]+)\.pdf$"

    # download each row
    for index, row in df_link.iterrows():
        # skip if downloaded already
        if df_link['downloaded'][index] == 0:
            match = re.search(pattern, row['link'])
            ## confirm all matches
            if not match:
                print("Unmatched row:")
                print(row)

            if match:
                substring = match.group(1)

                print(index)
                print(substring)
                # get request PDF and write to disk
                url = row['link']
                newfile = 'datastore/scrape/cr-bound/' + substring + ".pdf"
                download(url, newfile)
                df_link['downloaded'][index] = 1
                df_link.to_csv('datastore/scrape/cr-bound/link_list.csv', index = False) 

                ## check enough memory to proceed
                disk_usage = psutil.disk_usage('/')
                total_space = disk_usage.total
                # Convert bytes to gigabytes (GB)
                total_space_gb = total_space / (1024 ** 3)
                if total_space_gb <= 30:
                    print("System memory warning, 30 GB remains")
                    pdb.set_trace()


    return None

if __name__ == "__main__":
    Main()
