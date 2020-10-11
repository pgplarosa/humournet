"""
################################################################################
## Module Name: scraper.py
## Created by: Patrick La Rosa
## Created on: 30/09/2020
##
##
################################################################################
"""

# import libraries
from bs4 import BeautifulSoup as bs
import requests
import os
import pandas as pd
from tqdm import tqdm

# CONSTANTS
# number of pages to be used as image template
img_pages = 300

# number of pages to be used as captions
cap_pages = 3

# path to save images
save_path = 'images/'

# output filename to save dataset
out_fname = 'memes_dataset.csv'

# url site
url = 'https://memegenerator.net'

def scrape_memes(startpage=1):
  """scrape images and captions from memegenerator.net"""

  # initialize dataframe object
  df = pd.DataFrame(columns=['filename', 'caption'])

  # create directory if save_path does not exist
  if not os.path.exists(save_path):
    print(f'Created directory ./{save_path}')
    os.mkdir(save_path)

  # loop thru image pages
  for img_page in range(startpage, img_pages + 1):
    if img_page == 1:
      img_url = url + '/memes/popular/alltime/'
    else:
      img_url = url + '/memes/popular/alltime/page/' + str(img_page)
    
    print(f'Processing page {img_page}/{img_pages}..')

    resp = requests.get(img_url)
    soup = bs(resp.text, 'html.parser')
    chars = soup.find_all(class_='char-img')
    imgs_src = [char.find('img')['src'] for char in chars]
    img_links = [char.find('a')['href'] for char in chars]
    
    # loop thru each image templates in a page
    for img_link, img_src in tqdm(zip(img_links, imgs_src), position=0, leave=True):
      # get filename
      fname = img_src.split('/')[-1]
      complete_path = os.path.join(save_path, fname)
      
      # save image template (without annotations) in path
      resp = requests.get(img_src, stream=True)
      with open(complete_path,'wb') as img:
        img.write(resp.content)
      
      for cap_page in range(1, cap_pages + 1):
        if cap_page == 1:
          page_url = url + img_link
        else:
          page_url = (url + img_link + '/images/popular/alltime/page/' 
                      + str(cap_page))

        resp = requests.get(page_url)
        soup = bs(resp.text,'html.parser')
        caps = soup.find_all(class_='generator-img')
        cap_links = [cap.find('a')['href'] for cap in caps]

        # open image to get the actual captions
        for cap_link in cap_links:
          cap_url = url + cap_link
          resp = requests.get(cap_url)
          soup = bs(resp.text,'html.parser')
          caption = soup.find('title').string
          caption = caption[:caption.index('-')].strip()

          df = df.append(pd.DataFrame(data={'filename': [fname], 
                                            'caption': [caption]}), 
                                      ignore_index=True)

        # write to csv 
        if os.path.exists(out_fname):
          df.to_csv(out_fname, mode='a', index=False, header=False)
        else:
          df.to_csv(out_fname, index=False)

if __name__ == '__main__':
  scrape_memes()
