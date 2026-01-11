import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time

def search_and_download(title_query, filename_prefix):
    base_url = 'https://export.arxiv.org/api/query?' 
    search_query = f'ti:"{title_query}"'
    encoded_query = urllib.parse.quote(search_query)
    url = f'{base_url}search_query={encoded_query}&start=0&max_results=1'
    
    print(f"Searching for title: {title_query}")
    try:
        data = urllib.request.urlopen(url).read()
        root = ET.fromstring(data)
        
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        if not entries:
            print("No results found.")
            return

        entry = entries[0]
        id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
        arxiv_id = id_url.split('/abs/')[-1]
        found_title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
        
        print(f"Found: {found_title} ({arxiv_id})")
        
        pdf_link = ''
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if link.attrib.get('title') == 'pdf':
                pdf_link = link.attrib['href']
        
        if pdf_link:
            safe_title = "".join([c for c in found_title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]
            filename = f"papers/{filename_prefix}_{arxiv_id}_{safe_title}.pdf"
            
            if not os.path.exists(filename):
                print(f"Downloading to {filename}...")
                urllib.request.urlretrieve(pdf_link, filename)
                time.sleep(1) 
            else:
                print(f"File {filename} already exists.")

    except Exception as e:
        print(f"Error: {e}")

queries = [
    ("LoRA: Low-Rank Adaptation of Large Language Models", "lora"),
    ("Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", "intrinsic_dim"),
]

for q, prefix in queries:
    search_and_download(q, prefix)