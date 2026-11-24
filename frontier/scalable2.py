import json,re,time,urllib.request,urllib.parse,datetime as dt
since=(dt.date.today()-dt.timedelta(days=240)).isoformat()
Q=["BeyondMimic","humanoid motion tracking G1 mjlab","humanoid teleoperation whole-body imitation","humanoid motion retargeting","humanoid reinforcement learning framework Isaac Lab","humanoid general motion tracking","humanoid whole-body controller distillation","Unitree G1 sim-to-real reinforcement learning locomotion","humanoid foundation model motion generation control","motion tracking large-scale mocap dataset humanoid","humanoid loco-manipulation whole-body policy","universal humanoid controller","humanoid VLA whole-body control","SONIC humanoid","humanoid motion imitation adversarial AMP","legged robot locomotion foundation model","humanoid sim2real training framework MuJoCo Warp"]
G=lambda u:urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"fm"}),timeout=25).read().decode("utf8","ignore")
seen={r["repo"] for r in json.load(open("scalable.json"))}
c={}
for q in Q:
    try:
        for j in json.loads(G("https://api.github.com/search/repositories?per_page=10&sort=stars&order=desc&q="+urllib.parse.quote(f"{q} pushed:>{since} stars:>25")))["items"]:
            if j["full_name"] not in seen: c[j["full_name"]]=dict(repo=j["full_name"],stars=j["stargazers_count"],pushed=j["pushed_at"][:10],created=j["created_at"][:10],lic=(j.get("license") or {}).get("spdx_id"),desc=(j.get("description") or "")[:110],forks=j["forks_count"])
    except Exception as e: print("fail",q,e)
    time.sleep(2.5)
LISTY=re.compile(r"awesome|paper|survey|notes|handbook|daily|list|learning-?path|course",re.I)
out=[]
for r in sorted(c.values(),key=lambda x:-x["stars"]):
    if LISTY.search(r["repo"].split("/")[1]): continue
    if len(out)>=26: break
    try: t=G(f"https://raw.githubusercontent.com/{r['repo']}/HEAD/README.md")
    except Exception: t=""
    low=t
    r["arxiv"]=sorted(set(re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",t)))[:2]
    r["train"]=bool(re.search(r"train",t,re.I)) and bool(re.search(r"pip install|uv sync|conda|git clone",t,re.I))
    r["weights"]=bool(re.search(r"huggingface\.co/|checkpoint|pretrained",t,re.I))
    r["soon"]=bool(re.search(r"coming soon|to be released|will be released|\[ \]\s*\**(training )?code|☐",t,re.I))
    r["g1"]=bool(re.search(r"\bG1\b|Unitree",t))
    r["gpu"]=(re.findall(r"(?:single|one|\d+)[- ](?:RTX ?\d+|H100|A100|GPU)[^.\n]{0,40}",t)[:1])
    r["scale"]=(re.findall(r"[^.\n]{0,50}(?:\d[\d,\.]*\s*(?:hours|B parameters|M parameters|motions|clips|robots|embodiments|GPUs?))[^.\n]{0,40}",t)[:2])
    r["head"]=re.sub(r"\s+"," ",re.sub(r"[#*>\[\]!`|]|\(http[^)]*\)|<[^>]*>","",t))[:170]
    out.append(r)
json.dump(out,open("scalable2.json","w"),indent=1,ensure_ascii=False)
for r in out: print(r["stars"],r["pushed"],r["repo"],r["lic"],"train" if r["train"] else "-","W" if r["weights"] else "-","SOON" if r["soon"] else "","G1" if r["g1"] else "",r["arxiv"],"|",r["desc"][:60])
