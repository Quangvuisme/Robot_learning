#!/usr/bin/env python3
"""Verify candidate code repos against their paper by reading the repo README (raw.githubusercontent.com).
Output verified.json with a tier per (paper, repo). Only tiers A/B/R are email-worthy."""
import json,re,urllib.request,os,sys
HERE=os.path.dirname(os.path.abspath(__file__))
def readme(repo):
    for n in ("README.md","readme.md","README.MD","README.rst","README"):
        try: return urllib.request.urlopen(urllib.request.Request(f"https://raw.githubusercontent.com/{repo}/HEAD/{n}",headers={"User-Agent":"fm"}),timeout=20).read().decode("utf8","ignore")
        except Exception: pass
    return None
def norm(s): return re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
cands={}  # (arxiv id/title) -> list of (title,repo)
d=json.load(open(os.path.join(HERE,"data.json")))
for it in d["items"]:
    aid=it["url"].rsplit("/",1)[-1] if "arxiv.org" in it["url"] else None
    m=re.match(r"https://github\.com/([^/]+/[^/#?]+)",it.get("code",""))
    if m: cands.setdefault(it["title"],(aid,[]))[1].append(m[1].removesuffix(".git"))
    g=it.get("gh")
    if g: cands.setdefault(it["title"],(aid,[]))[1].append(g["full"])
for p in json.load(open(os.path.join(HERE,"repro.json"))):
    aid=p["url"].rsplit("/",1)[-1]
    for r in p["repos"]: cands.setdefault(p["title"],(aid,[]))[1].append(r["repo"])
for extra in json.load(open(os.path.join(HERE,"extra_repos.json"))) if os.path.exists(os.path.join(HERE,"extra_repos.json")) else []:
    cands.setdefault(extra["title"],(extra.get("arxiv"),[]))[1].append(extra["repo"])
LISTY=re.compile(r"awesome|arxiv|daily|paper-?list|papers|digest|survey|handbook|watch|notes|github\.io|atlas|reading",re.I)
OFFICIAL=re.compile(r"official (implementation|code|repo|repository|release)|this is the official|the official (code|implementation)|official pytorch",re.I)
UNOFF=re.compile(r"unofficial|re-?implementation of|(independent|our|a|faithful) (re)?production of|reproduction of|independent(ly)? (implementation|reproduc)|not affiliated|community (port|implementation)",re.I)
SOON=re.compile(r"[☐□]\s*\**(training )?code|\[ \]\s*\**(training )?code|code,? including inference|being prepared|currently contains the editable|coming soon|code (will be|to be) (released|made)|to be released|release soon|stay tuned|todo.*(code|release)|\[ \].*(code|release)",re.I)
USAGE=re.compile(r"pip install|conda (create|env)|git clone|python[3]? .*\.py|uv (sync|pip)|docker (run|build)|## ?(installation|getting started|usage|quick ?start|training|evaluation)",re.I)
WEIGHTS=re.compile(r"huggingface\.co/|checkpoint|pretrained|weights|\.pt\b|\.ckpt",re.I)
out=[]
seen=set()
for title,(aid,repos) in cands.items():
    for repo in dict.fromkeys(repos):
        if (title,repo) in seen: continue
        seen.add((title,repo))
        rec=dict(title=title,arxiv=aid,repo=repo)
        txt=readme(repo)
        if txt is None: rec.update(tier="X",why="no README or private"); out.append(rec); continue
        low=txt.lower(); nt=norm(txt)
        rec["id_in_readme"]=bool(aid and aid in txt)
        name=re.split(r"[:\-]",title)[0].strip().lower()
        rec["title_in_readme"]=norm(title)[:50] in nt
        rec["name_in_readme"]=bool(name and len(name)>3 and name in low)
        rec["official_claim"]=bool(OFFICIAL.search(txt)); rec["unofficial_claim"]=bool(UNOFF.search(txt))
        rec["coming_soon"]=bool(SOON.search(txt)); rec["usage"]=bool(USAGE.search(txt)); rec["weights"]=bool(WEIGHTS.search(txt))
        rec["len"]=len(txt)
        ev=[]
        for nm,rx in(("official",OFFICIAL),("unofficial",UNOFF),("soon",SOON)):
            m=rx.search(txt)
            if m: ev.append(nm+": ..."+txt[max(0,m.start()-60):m.end()+60].replace("\n"," ")+"...")
        rec["evidence"]=ev
        rec["head"]=re.sub(r"\s+"," ",re.sub(r"[#*>\[\]!`|]|\(http[^)]*\)|<[^>]+>","",txt))[:220]
        about=rec["id_in_readme"] or rec["title_in_readme"] or rec["name_in_readme"]
        if LISTY.search(repo.split("/")[1]): rec.update(tier="L",why="list/site repo")
        elif not about: rec.update(tier="X",why="README does not mention the paper")
        elif rec["unofficial_claim"] and not rec["official_claim"]: rec.update(tier="R" if rec["usage"] and not rec["coming_soon"] else "S",why="reproduction/unofficial" if rec["usage"] else "unofficial, no usage")
        elif rec["coming_soon"] and not rec["usage"]: rec.update(tier="S",why="code coming soon")
        elif rec["coming_soon"] and not rec["usage"]: rec.update(tier="S",why="code not released yet")
        elif rec["usage"] and (rec["id_in_readme"] or rec["title_in_readme"]) and rec["official_claim"]: rec.update(tier="A",why="paper cited + official claim + runnable usage")
        elif rec["usage"] and (rec["id_in_readme"] or rec["title_in_readme"]): rec.update(tier="B",why="paper cited + runnable usage, no explicit 'official' claim")
        else: rec.update(tier="S",why="mentions paper but no usage instructions")
        out.append(rec)
json.dump(out,open(os.path.join(HERE,"verified.json"),"w"),indent=1)
from collections import Counter; print(Counter(r["tier"] for r in out))
for r in out:
    if r["tier"] in "ABR": print(r["tier"],r["repo"],"|",r["title"][:45],"|",r["why"],"| weights" if r.get("weights") else "")
