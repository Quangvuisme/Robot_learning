#!/usr/bin/env python3
"""Find repos for scalable / multi-skill / continual humanoid whole-body control, then read each README."""
import json,re,time,urllib.request,urllib.parse,datetime as dt
since=(dt.date.today()-dt.timedelta(days=150)).isoformat()
Q=["humanoid behavior foundation model","humanoid motion tracking multi-skill","humanoid whole-body control mixture of experts","humanoid continual learning skills","humanoid universal motion tracker","humanoid whole-body control foundation model pretrained","humanoid general motion tracking sim-to-real","humanoid skill library adapter LoRA","whole-body tracking scalable training thousands of motions","humanoid cross-embodiment whole-body control"]
G=lambda u:urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"fm"}),timeout=25).read().decode("utf8","ignore")
c={}
for q in Q:
    try:
        for j in json.loads(G("https://api.github.com/search/repositories?per_page=10&sort=stars&order=desc&q="+urllib.parse.quote(f"{q} pushed:>{since} stars:>15")))["items"]:
            c[j["full_name"]]=dict(repo=j["full_name"],stars=j["stargazers_count"],pushed=j["pushed_at"][:10],created=j["created_at"][:10],lic=(j.get("license") or {}).get("spdx_id"),desc=(j.get("description") or "")[:100])
    except Exception as e: print("fail",q,e)
    time.sleep(2.5)
KW={"moe":r"mixture.of.experts|\bMoE\b|router|plug-?in|adapter|LoRA","multi":r"multi-?skill|general (motion )?track|universal|foundation model|behavior foundation|large-scale|scal(e|able|ing)","weights":r"huggingface\.co/|checkpoint|pretrained","usage":r"pip install|conda (create|env)|uv sync|git clone","train":r"\btrain(ing)?\b.*(script|command|python)|python .*train","soon":r"coming soon|code (will be|to be) released|\[ \]\s*\**(training )?code","sim2real":r"sim-?to-?real|real[- ]world|real robot|deploy"}
out=[]
for r in sorted(c.values(),key=lambda x:-x["stars"])[:32]:
    try: t=G(f"https://raw.githubusercontent.com/{r['repo']}/HEAD/README.md")
    except Exception: t=""
    r["flags"]=[k for k,rx in KW.items() if re.search(rx,t,re.I)]
    r["head"]=re.sub(r"\s+"," ",re.sub(r"[#*>\[\]!`|]|\(http[^)]*\)|<[^>]*>","",t))[:200]
    r["arxiv"]=sorted(set(re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",t)))[:2]
    out.append(r)
json.dump(out,open("scalable.json","w"),indent=1,ensure_ascii=False)
for r in out: print(r["stars"],r["pushed"],r["repo"],r["lic"],r["flags"],r["arxiv"],"|",r["desc"][:70])
