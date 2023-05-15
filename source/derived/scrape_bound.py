import os
import pandas as pd
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
import pdb
import time
import re


### downloads PDFs of the Congressional Record (bound version)
def Main():
    raw_dir = "datastore/raw/cr-bound"

    link_list = GetLinks(raw_dir)
    DownloadPDFs(link_list)

# gets a list of links to Congressional Record bound directories to download
def GetLinks(infolder):

    if not os.path.isfile('datastore/raw/cr-bound/link_list.csv'):
        # set up webdriver 
        chrome_options =ChromeOptions()
        # chrome_options.add_argument('--headless')
        chrome_options.add_experimental_option('w3c', True)
        driver = Chrome('datastore/chromedriver_mac64/chromedriver', options=chrome_options)

        # get web page
        driver.get("https://www.govinfo.gov/app/collection/crecb")
        time.sleep(10) # seconds


        # Loop through all the panel headings and click them to expand the panels
        panel_headings = driver.find_elements(by=By.CLASS_NAME, value="panel-heading")
        for panel_heading in panel_headings:
            time.sleep(1) # seconds
            panel_heading.click()

        # get href of buttons
        buttons = driver.find_elements(by=By.CLASS_NAME, value="btn")
        links = [button.get_attribute('href') for button in buttons]
        # only want pdf
        link_list = [l for l in links if l is not None and ".pdf" in l]

        # write to disk
        df_link = pd.DataFrame({'link': link_list})
        df_link.to_csv('datastore/raw/cr-bound/link_list.csv') 

    return 'datastore/raw/cr-bound/link_list.csv'

# downloads list of links
def DownloadPDFs(link_list):

    ## TODO: write function
    df_link = pd.read_csv(link_list)

    # between last backslash and pdf
    pattern = r"/([^/]+)\.pdf$"

    for index, row in df.iterrows():
        match = re.search(pattern, row['link'])
        if match:
            substring = match.group(1)
            print(substring)

    return None

if __name__ == "__main__":
    Main()
