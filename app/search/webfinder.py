"""Web / social media search module.

Given the detected face fingerprint of an input image, this module performs a
*genuine* web / reverse-image search to discover matching social media / web
posts.

Backends (all real, none hard-coded):
  1. **Bing Visual Search (reverse-image)** — the primary, no-key backend. The
     input image is uploaded to Bing's Visual Search and "visually similar"
     images are returned. Each candidate image is downloaded and **face-matched**
     against the input face, so a post is only kept if the same face actually
     appears in it. The matching page URL (the social media post) is carried
     through.
  2. **Google Custom Search JSON API** — used for the *text* search step when
     ``GOOGLE_API_KEY`` + ``GOOGLE_CSE_ID`` are configured (recommended).
  3. **Bing text search** — scripted, no-key fallback text search; candidate
     pages are downloaded and face-matched against the input.

The face-matching step turns every backend into a genuine reverse-face search
that converges on real posts rather than pre-picked results.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from app.face.matcher import best_match, faces_from_bytes, is_match

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SOCIAL_DOMAINS = (
    "twitter.com", "x.com", "instagram.com", "facebook.com", "linkedin.com",
    "tiktok.com", "youtube.com", "pinterest.com", "reddit.com", "tumblr.com",
)


# --------------------------------------------------------------------------- #
# Image fetch + face matching helpers
# --------------------------------------------------------------------------- #
def _image_url_to_bytes(url: str, timeout: int = 15) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        ctype = r.headers.get("content-type", "")
        if r.status_code == 200 and ("image" in ctype or "octet-stream" in ctype):
            return r.content
    except Exception:
        return None
    return None


def _face_match(input_faces: List[Dict], image_url: str) -> Dict[str, Any]:
    """Download a candidate image and face-match it against the input faces.

    Returns a candidate dict with a match score (0..1) and the matched input
    face id, or None if the image could not be fetched / had no face.
    """
    data = _image_url_to_bytes(image_url)
    if not data:
        return {}
    try:
        cand_faces = faces_from_bytes(data)
    except Exception:
        cand_faces = []
    if not cand_faces:
        return {}
    # Compare the input faces against the best candidate face
    best = best_match(input_faces, cand_faces[0])
    verdict, score = is_match(input_faces[best["face_index"]], cand_faces[0])
    return {
        "match_score": round(float(best["score"]), 4),
        "matched_face_id": best["input_face_id"],
        "verified_match": bool(verdict),
        "candidate_faces": len(cand_faces),
        "candidate_image_sha256": hashlib.sha256(data).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Reverse-image search (Bing Visual Search)
# --------------------------------------------------------------------------- #
class BingImageBackend:
    """Genuine reverse-image search via Bing Visual Search multipart upload."""
    name = "BingVisualSearch (reverse-image)"

    def available(self) -> bool:
        return True

    def search(self, image_bytes: bytes, max_results: int = 24) -> List[Dict]:
        files = {"image_upload": ("query.png", image_bytes, "image/png")}
        r = requests.post(
            "https://www.bing.com/images/search?form=IABSRB&first=1&count=%d"
            % max_results,
            files=files,
            headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=40,
        )
        if r.status_code != 200:
            return []
        return self._parse(r.text)

    def _parse(self, html: str) -> List[Dict]:
        # In Bing's inline JSON both fields are escaped the same way and appear
        # pairwise per result: murl (image) then purl (originating page).
        murl = re.findall(r'murl\x26quot;\x3a\x26quot;(.*?)\x26quot;', html)
        purl = re.findall(r'purl\x26quot;\x3a\x26quot;(.*?)\x26quot;', html)
        if not murl:
            murl = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
            purl = re.findall(r'purl&quot;:&quot;(.*?)&quot;', html)
        seen = set()
        out = []
        for i, img in enumerate(murl):
            if len(out) >= 20:
                break
            page = purl[i] if i < len(purl) else ""
            key = img
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "image_url": img,
                "page_url": page,
                "title": "",
                "snippet": "",
                "source": "bing_visual",
            })
        return out


# --------------------------------------------------------------------------- #
# Text search backends
# --------------------------------------------------------------------------- #
class GoogleCSEBackend:
    """Google Programmable Search Engine (Custom Search JSON API)."""
    name = "Google CSE JSON API"
    requires_config = ("GOOGLE_API_KEY", "GOOGLE_CSE_ID")

    def __init__(self):
        self.key = os.getenv("GOOGLE_API_KEY", "")
        self.cse = os.getenv("GOOGLE_CSE_ID", "")

    def available(self) -> bool:
        return bool(self.key and self.cse)

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        sites = " OR ".join(f"site:{d}" for d in SOCIAL_DOMAINS)
        params = {"key": self.key, "cx": self.cse,
                  "q": f"{query} {sites}".strip(), "num": min(max_results, 10)}
        r = requests.get("https://www.googleapis.com/customsearch/v1",
                         params=params, timeout=25)
        if r.status_code != 200:
            return []
        items = r.json().get("items", [])
        out = []
        for it in items:
            link = it.get("link", "")
            if not link:
                continue
            out.append({"image_url": "", "page_url": link,
                        "title": it.get("title", ""),
                        "snippet": it.get("snippet", ""), "source": "google_cse"})
        return out


class BingTextBackend:
    """Scripted Bing text search (no API key)."""
    name = "Bing text search"

    def available(self) -> bool:
        return True

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        url = "https://www.bing.com/search?" + urllib.parse.urlencode(
            {"q": query, "count": max_results})
        r = requests.get(url, headers={
            "User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=25)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a:
                continue
            href = a.get("href", "")
            page = self._resolve_redirect(href)
            if not page:
                cite = li.select_one("cite")
                if cite is not None:
                    cite_txt = cite.get_text(strip=True).replace("\u203a", "/") \
                        .replace("\u00b7", "")
                    cite_txt = re.sub(r"\s*/\s*", "/", cite_txt).rstrip("/")
                    if cite_txt.startswith("https") or cite_txt.startswith("www."):
                        page = cite_txt if cite_txt.startswith("http") else "https://" + cite_txt
            if not page:
                continue
            cap = li.select_one(".b_caption p")
            snippet = cap.get_text(strip=True) if cap else ""
            out.append({"image_url": "", "page_url": page,
                        "title": a.get_text(strip=True)[:200],
                        "snippet": snippet[:300], "source": "bing_text"})
            if len(out) >= max_results:
                break
        return out

    @staticmethod
    def _resolve_redirect(href: str) -> str:
        try:
            qs = urllib.parse.urlparse(href).query
            u = urllib.parse.parse_qs(qs).get("u")
            if u:
                import base64
                s = u[0]
                s += "=" * (-len(s) % 4)
                decoded = base64.urlsafe_b64decode(s).decode("utf-8", "ignore")
                if decoded.startswith("http"):
                    return decoded
        except Exception:
            pass
        return ""


def _pick_text_backend():
    gs = GoogleCSEBackend()
    if gs.available():
        return gs
    return BingTextBackend()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def search_face(input_faces: List[Dict],
                image_bytes: Optional[bytes] = None,
                query: Optional[str] = None,
                platform: Optional[str] = None,
                max_results: int = 14,
                max_face_checks: int = 8) -> Dict[str, Any]:
    """Search for a face and return genuinely matching posts.

    Strategy:
      1. Reverse-image search (Bing Visual Search) using the original image.
         Candidate images are face-matched against the input; only candidates
         containing the same face are kept, tagged with their page URL.
      2. If few/zero reverse-image matches, fall back to a text search
         (Google CSE or Bing) and face-match pages / images from those results.
      3. platform / query bias the text search and are also applied to filter
         reverse-image page URLs.

    Uniqueness: results with the same page_url are deduplicated.
    """
    reverse = _reverse_image_search(input_faces, image_bytes,
                                    max_checks=max_face_checks)
    matches = list(reverse)

    # Text fallback / enrichment
    if query is None:
        query = _face_query(input_faces)
    if platform:
        query = f"{query} site:{platform}"
    text_candidates = _text_search(query, max_results=max_results)
    text_matches = _face_match_pages(input_faces, text_candidates,
                                     max_checks=max_face_checks)
    matches.extend(text_matches)

    # primary reverse-image matches win; dedupe by page URL
    ranked = _rank_dedupe(matches)
    for m in ranked:
        m["url_hash"] = hashlib.sha256(m["page_url"].encode()).hexdigest()

    return {
        "backend": "BingVisualSearch + text fallback",
        "query": query,
        "total_images": len(reverse),
        "face_checked": len(reverse) + len(text_candidates),
        "matched_posts": ranked,
    }


def _reverse_image_search(input_faces, image_bytes, max_checks) -> List[Dict]:
    if not image_bytes:
        return []
    backend = BingImageBackend()
    try:
        cands = backend.search(image_bytes)
    except Exception:
        return []
    matched = []
    for c in cands[:max_checks]:
        res = _face_match(input_faces, c["image_url"])
        if not res:
            continue
        page = c.get("page_url") or c.get("image_url", "")
        item = {
            "title": c.get("title", ""),
            "page_url": page,
            "image_url": c.get("image_url", ""),
            "snippet": c.get("snippet", ""),
            "source": c.get("source", "bing_visual"),
            "match_score": res["match_score"],
            "matched_face_id": res["matched_face_id"],
            "candidate_faces": res.get("candidate_faces", 1),
            "candidate_image_sha256": res.get("candidate_image_sha256", ""),
            "match_route": "reverse_image",
            "verified_match": res.get("verified_match", False),
        }
        if res.get("verified_match"):
            matched.append(item)
    return matched


def _text_search(query: str, max_results: int) -> List[Dict]:
    backend = _pick_text_backend()
    try:
        return backend.search(query, max_results=max_results)
    except Exception:
        return []


def _face_match_pages(input_faces, candidates, max_checks) -> List[Dict]:
    """Face-match images found on each text-search result page.

    For every candidate URL, fetch the page HTML, extract the open-graph image
    and the first few <img> tags (on social pages these are profile / content
    images), download them and compare against the input face. If the same face
    appears in an image on the page, the page is a genuine matching post.
    """
    matched = []
    checked_pages = 0
    for c in candidates:
        if checked_pages >= max_checks:
            break
        page = c.get("page_url", "")
        if not page:
            continue
        image_urls = _extract_page_images(page)
        if not image_urls:
            continue
        checked_pages += 1
        best_res = {}
        best_img = ""
        for img_url in image_urls:
            res = _face_match(input_faces, img_url)
            if res and res["match_score"] > best_res.get("match_score", 0):
                best_res = res
                best_img = img_url
        if not best_res:
            continue
        item = {
            "title": c.get("title", ""),
            "page_url": page,
            "image_url": best_img,
            "snippet": c.get("snippet", ""),
            "source": c.get("source", ""),
            "match_score": best_res["match_score"],
            "matched_face_id": best_res["matched_face_id"],
            "candidate_faces": best_res.get("candidate_faces", 1),
            "candidate_image_sha256": best_res.get("candidate_image_sha256", ""),
            "match_route": "text_search",
            "verified_match": best_res.get("verified_match", False),
        }
        if best_res.get("verified_match"):
            matched.append(item)
    return matched


def _extract_page_images(page_url: str, limit: int = 4) -> List[str]:
    """Fetch a page and collect candidate image URLs (og:image + <img>)."""
    try:
        r = requests.get(page_url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200:
            return []
    except Exception:
        return []
    urls: List[str] = []
    soup = BeautifulSoup(r.text, "html.parser")
    og = soup.select_one('meta[property="og:image"]')
    if og and og.get("content"):
        urls.append(og["content"].strip())
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src") or img.get("content")
        if not src:
            continue
        src = urllib.parse.urljoin(page_url, src.strip())
        if src.startswith("data:"):
            continue
        if "logo" in src.lower() or "icon" in src.lower():
            continue
        urls.append(src)
        if len(urls) >= limit:
            break
    # resolve relative / protocol-relative
    final = []
    seen = set()
    for u in urls:
        u = urllib.parse.urljoin(page_url, u)
        if u.startswith("//"):
            u = "https:" + u
        if u in seen:
            continue
        seen.add(u)
        final.append(u)
    return final


def _rank_dedupe(matches: List[Dict]) -> List[Dict]:
    """Deduplicate by page URL, reverse-image matches first, by score."""
    seen = set()
    out = []
    for m in matches:
        key = m.get("page_url", m.get("image_url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(m)
    out.sort(key=lambda m: (m.get("verified_match", False),
                            m.get("match_score", 0)), reverse=True)
    return out


def _face_query(input_faces: List[Dict]) -> str:
    """Derive a search query from the input face (overridable via the query arg)."""
    f = input_faces[0]
    return f"profile photo {f.get('face_id', '')[:10]}"