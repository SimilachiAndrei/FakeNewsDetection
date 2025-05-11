import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time


def get_articles_from_digi24_search(page):
    base_url = f"https://www.digi24.ro/cautare?q=alegeri+prezidentiale&ps=10&p={page}"
    print(f"\n[PAGE {page}] Accesez: {base_url}")
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    results = soup.find_all("h2", class_="h4 article-title")
    print(f"[INFO] Am găsit {len(results)} articole pe pagina de căutare.")

    for idx, h2 in enumerate(results, start=1):
        a_tag = h2.find("a")
        if not a_tag:
            print(f"[WARN] Articolul #{idx} nu are link.")
            continue
        link = "https://www.digi24.ro" + a_tag["href"]
        headline = a_tag.text.strip()

        print(f"[ARTICLE {idx}] Titlu: {headline}")
        print(f"[ARTICLE {idx}] Intru pe link: {link}")

        # Accesăm articolul
        try:
            article_resp = requests.get(link)
            article_resp.raise_for_status()
            article_soup = BeautifulSoup(article_resp.text, "html.parser")
            print(f"[ARTICLE {idx}] ✔️ Accesat cu succes")
        except Exception as e:
            print(f"[ARTICLE {idx}] ❌ Eroare la accesare: {e}")
            continue

        # Extragem conținutul
        content_div = article_soup.find("div", class_="entry data-app-meta data-app-meta-article")
        content = content_div.get_text(strip=True, separator=' ') if content_div else ""
        if not content:
            print(f"[ARTICLE {idx}] ⚠️ Conținut gol sau lipsă.")

        # Extragem data publicării
        author_meta = article_soup.find("div", class_="author-meta")
        if author_meta:
            time_tag = author_meta.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                date = time_tag["datetime"]
            else:
                date = str(datetime.today().date())
                print(f"[ARTICLE {idx}] ⚠️ Tag <time> fără atribut datetime.")
        else:
            date = str(datetime.today().date())
            print(f"[ARTICLE {idx}] ⚠️ Lipsă <div class='author-meta'>.")

        keywords = ",".join({word.lower() for word in headline.split() if len(word) > 4})

        articles.append({
            "id": len(articles) + 1,
            "headline": headline,
            "keywords": keywords,
            "contents": content,
            "date": date,
            "source": link
        })

        time.sleep(0.5)  # pauză mică

    return articles

# Parcurgem paginile
all_articles = []

# AICI DOMNE ALEGI CATE PAGINI VREI. fiecare are cate 10 articole
for page_num in range(1, 4):  # ajustează numărul de pagini după nevoie
    articles = get_articles_from_digi24_search(page_num)
    all_articles.extend(articles)
    time.sleep(1)

# Salvare în Excel
df = pd.DataFrame(all_articles)
df["id"] = range(1, len(df) + 1)
df.to_excel("stiri_digi24_alegeri.xlsx", index=False)
print(f"\n[FINAL] Am salvat {len(df)} articole în fișierul stiri_digi24_alegeri.xlsx.")
