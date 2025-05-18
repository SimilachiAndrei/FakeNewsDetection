import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

BASE_URL = "https://adevarul.ro"
START_URL = f"{BASE_URL}/alegeri-prezidentiale-2025"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_articles_from_adevarul():
    print(f"[START] Accesez: {START_URL}")
    try:
        response = requests.get(START_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Eroare la accesarea paginii principale: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    article_links = soup.select("h2 a[href^='/alegeri-prezidentiale-2025/']")  # Selector pentru articole

    print(f"[INFO] Găsite {len(article_links)} articole")
    articles = []

    for idx, a_tag in enumerate(article_links, start=1):
        href = a_tag.get("href")
        link = href if href.startswith("http") else f"{BASE_URL}{href}"
        headline = a_tag.get_text(strip=True)

        try:
            art_resp = requests.get(link, headers=HEADERS, timeout=10)
            art_resp.raise_for_status()
        except Exception:
            continue

        art_soup = BeautifulSoup(art_resp.text, "html.parser")
        content_div = art_soup.find("div", class_="article-body")
        content = content_div.get_text(separator=" ", strip=True) if content_div else ""

        # Extragere dată publicare
        date = None
        date_meta = art_soup.find("meta", {"property": "article:published_time"})
        if date_meta and date_meta.has_attr("content"):
            try:
                date_obj = datetime.fromisoformat(date_meta["content"])
                date = date_obj.date()
            except:
                pass

        keywords = ",".join({w.lower() for w in headline.split() if len(w) > 4})

        print(f"✔️ {headline}")
        print(f"   ↪️ {link}")

        articles.append({
            "id": idx,
            "headline": headline,
            "keywords": keywords,
            "contents": content,
            "date": date,
            "source": link
        })

        time.sleep(0.5)

    return articles

if __name__ == "__main__":
    all_articles = get_articles_from_adevarul()
    df = pd.DataFrame(all_articles)
    df.drop_duplicates(subset="source", inplace=True)
    df["id"] = range(1, len(df) + 1)
    df.to_csv("../data/stiri_adevarul_alegeri.csv", index=False)
    print(f"\n✅ Am salvat {len(df)} articole în ../data/stiri_adevarul_alegeri.csv")
