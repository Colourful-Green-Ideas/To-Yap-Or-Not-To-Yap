import os
import re
import time
import html
import requests


# List of authors
AUTHORS = [
    "प्रेमचंद",
    "जयशंकर प्रसाद",
    "देवकीनन्दन खत्री",
    "किशोरीलाल गोस्वामी",
    "विश्वम्भरनाथ शर्मा कौशिक",
    "आचार्य चतुरसेन",
    "चन्द्रधर शर्मा गुलेरी",
    "शिवपूजन सहाय",
    "राहुल सांकृत्यायन",
    "गोपालराम गहमरी",
    "बालकृष्ण भट्ट",
    "भारतेन्दु हरिश्चन्द्र",
]

MAX_FILES_PER_AUTHOR = 250
MIN_WORDS = 500

API = "https://hi.wikisource.org/w/api.php"
OUT = "raw/hi_fiction"

# Create a session for API requests
session = requests.Session()
# Wikimedia asks scripts to include a contact email here
session.headers["User-Agent"] = "CL3202-student-project/1.0 (rennysharma29@gmail.com)"


# Send a request to Wikisource
def api(**params):
    params["format"] = "json"
    params["formatversion"] = 2

    for attempt in range(8):
        try:
            response = session.get(API, params=params, timeout=30)

            # 429 = we are going too fast; wait as long as Wikisource asks
            if response.status_code == 429:
                wait = response.headers.get("Retry-After", "60")

                if wait.isdigit():
                    wait = int(wait)
                else:
                    wait = 60

                print("Too many requests, waiting", wait, "seconds...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            time.sleep(1)    # 1 request per second
            return response.json()

        except Exception as e:
            print("Network error, trying again:", e)
            time.sleep(10)

    return {}


# Find the author namespace
def author_namespace():
    data = api(action="query", meta="siteinfo", siprop="namespaces")

    for ns in data.get("query", {}).get("namespaces", {}).values():
        if ns.get("canonical") == "Author" or ns.get("name") == "लेखक":
            return ns["id"]

    return 102


# Find an author's page
def find_author_page(name, ns):
    data = api(
        action="query",
        list="search",
        srsearch=name,
        srnamespace=ns,
        srlimit=1
    )

    results = data.get("query", {}).get("search", [])

    if len(results) > 0:
        return results[0]["title"]

    return None


# Get all works linked to an author
def works_of(author_page):
    titles = []

    params = {
        "action": "query",
        "prop": "links",
        "titles": author_page,
        "plnamespace": 0,
        "pllimit": "max"
    }

    while True:
        data = api(**params)

        pages = data.get("query", {}).get("pages", [])

        for page in pages:
            for link in page.get("links", []):
                titles.append(link["title"])

        if "continue" not in data:
            break

        params.update(data["continue"])

    return titles


# Find chapters or parts of a book
def subpages_of(title):
    data = api(
        action="query",
        list="allpages",
        apprefix=title + "/",
        apnamespace=0,
        aplimit="max"
    )

    pages = data.get("query", {}).get("allpages", [])

    titles = []

    for page in pages:
        titles.append(page["title"])

    return titles


# Get the text from a page
def page_text(title):
    data = api(action="parse", page=title, prop="text", redirects=1)

    page_html = data.get("parse", {}).get("text", "")

    # Remove page numbers and footnotes
    page_html = re.sub(
        r'<span[^>]*pagenum.*?</span>',
        " ",
        page_html,
        flags=re.S
    )

    page_html = re.sub(r'<sup.*?</sup>', " ", page_html)
    page_html = re.sub(r'<br\s*/?>', "\n", page_html)

    paragraphs = []

    # Extract paragraphs
    found_paragraphs = re.findall(r"<p>(.*?)</p>", page_html, flags=re.S)

    for p in found_paragraphs:
        text = re.sub(r"<[^>]+>", "", p)
        text = html.unescape(text).strip()

        if text != "":
            paragraphs.append(text)

    return paragraphs


# Check if the text looks like prose
def looks_like_prose(paragraphs):
    words = 0
    lines = 0

    for p in paragraphs:
        words += len(p.split())
        lines += p.count("\n") + 1

    if words < MIN_WORDS:
        return False

    if words / max(lines, 1) < 20:
        return False

    return True


# Make a safe filename
def safe_filename(title):
    title = re.sub(r'[\\/:*?"<>|]', "_", title)
    return title[:150] + ".txt"


# Main program

ns = author_namespace()
summary = {}

for name in AUTHORS:

    author_page = find_author_page(name, ns)

    if author_page is None:
        print(name, ": no author page found")
        continue

    author = author_page.split(":", 1)[-1]

    folder = os.path.join(OUT, author)
    os.makedirs(folder, exist_ok=True)

    works = works_of(author_page)

    print("\n", author, ":", len(works), "works found")

    saved = len(os.listdir(folder))

    for work in works:

        if saved >= MAX_FILES_PER_AUTHOR:
            break

        # Get chapters if the work has them
        pages = subpages_of(work)

        if len(pages) == 0:
            pages = [work]

        for title in pages:

            if saved >= MAX_FILES_PER_AUTHOR:
                break

            path = os.path.join(folder, safe_filename(title))

            # Skip files that already exist
            if os.path.exists(path):
                continue

            paragraphs = page_text(title)

            if not looks_like_prose(paragraphs):
                continue

            # Save the text
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(paragraphs))

            saved += 1
            print("[", saved, "]", title)

    summary[author] = saved


# Remove empty folders
for author in os.listdir(OUT):
    folder = os.path.join(OUT, author)

    # Skip hidden Mac files like .DS_Store
    if not os.path.isdir(folder):
        continue

    if len(os.listdir(folder)) == 0:
        os.rmdir(folder)


# Print the final count
print("\n===== files per author =====")

for author, count in summary.items():
    print(count, author)

total = sum(summary.values())

print(total, "TOTAL")