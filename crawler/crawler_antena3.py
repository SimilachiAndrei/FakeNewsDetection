import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re

BASE_URL = "https://www.antena3.ro"
HEADERS = {"User-Agent": "Mozilla/5.0"}

KEYWORDS = [
    "alegeri", "alegerile", "vot", "votanți", "candidat", "candidați", "candidatură",
    "campanie", "electoral", "președinte", "prezidențial", "prezidențiale", "bec",
    "ccr", "exit-poll", "turul doi", "turul 2", "urna", "dezbatere", "program electoral"
]
CANDIDATI = ["georgescu", "simion", "nicușor", "ciuacalu", "bolo", "burduja", "șoșoacă"]

def get_articles_from_politica_page(page, seen_links):
    url = f"{BASE_URL}/politica/p{page}.html" if page > 1 else f"{BASE_URL}/politica/"
    print(f"\n[PAGE {page}] Accesez: {url}")

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Eroare la accesare pagină: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles_html = soup.find_all("article")
    print(f"[INFO] Găsite {len(articles_html)} articole")

    articles = []

    for article in articles_html:
        a_tag = article.find("a", href=True)
        if not a_tag:
            continue

        link = a_tag["href"]
        if not link.startswith("http"):
            link = BASE_URL + link

        # evităm duplicarea
        if link in seen_links:
            continue
        seen_links.add(link)

        headline = a_tag.get_text(strip=True)

        try:
            art_resp = requests.get(link, headers=HEADERS)
            art_resp.raise_for_status()
            art_soup = BeautifulSoup(art_resp.text, "html.parser")
        except Exception:
            continue

        content_div = art_soup.find("div", class_="article-content")
        content = content_div.get_text(separator=" ", strip=True) if content_div else ""

        # extragere dată
        date = None
        meta_time = art_soup.find("meta", {"property": "article:published_time"})
        if meta_time and meta_time.has_attr("content"):
            try:
                date_obj = datetime.fromisoformat(meta_time["content"])
                date = date_obj.date()
            except:
                pass
        else:
            date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', content)
            if date_match:
                try:
                    date_obj = datetime.strptime(date_match.group(1), "%d.%m.%Y")
                    date = date_obj.date()
                except:
                    pass

        # ignorăm dacă nu e în intervalul ales
        if date and not (datetime(2024, 11, 1).date() <= date <= datetime(2025, 5, 17).date()):
            continue

        full_text = f"{headline.lower()} {content.lower()}"
        if not any(kw in full_text for kw in KEYWORDS + CANDIDATI):
            continue

        keywords = ",".join({w.lower() for w in headline.split() if len(w) > 4})

        print(f"✔️ {headline}")
        print(f"   ↪️ {link}")

        articles.append({
            "id": len(seen_links),  # provizoriu
            "headline": headline,
            "keywords": keywords,
            "contents": content,
            "date": date,
            "source": link
        })

        time.sleep(0.5)

    return articles


if __name__ == "__main__":
    all_articles = []
    seen_links = set()

    for page_num in range(1, 4):  # ajustează după nevoie
        articles = get_articles_from_politica_page(page_num, seen_links)
        all_articles.extend(articles)
        time.sleep(1)

    df = pd.DataFrame(all_articles)
    df["id"] = range(1, len(df) + 1)
    df.to_csv("../data/stiri_antena3_alegeri.csv", index=False)
    print(f"\n✅ Am salvat {len(df)} articole în ../data/stiri_antena3_alegeri.csv")
