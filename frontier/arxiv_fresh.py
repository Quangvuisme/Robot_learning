#!/usr/bin/env python3
"""Fresh papers straight from arXiv (export API) + Hugging Face daily-papers signals (upvotes, github repo)."""
import json,re,urllib.request,urllib.parse,datetime as dt,xml.etree.ElementTree as ET,os,time
HERE=os.path.dirname(os.path.abspath(__file__))
G=lambda u:urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"}),timeout=30).read().decode()
terms=['humanoid','locomotion','"whole-body"','"vision-language-action"','"world action model"','"legged"','"motion tracking"','"dexterous"','"robot foundation model"','"sim-to-real"','"loco-manipulation"','teleoperation']
q="(cat:cs.RO OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV) AND ("+" OR ".join("abs:"+t if " " not in t else "abs:"+t for t in terms)+")"
u="https://export.arxiv.org/api/query?"+urllib.parse.urlencode(dict(search_query=q,sortBy="submittedDate",sortOrder="descending",max_results=120))
ns={"a":"http://www.w3.org/2005/Atom"}
root=ET.fromstring(G(u)); papers=[]
for e in root.findall("a:entry",ns):
    aid=re.sub(r"v\d+$","",e.find("a:id",ns).text.rsplit("/",1)[-1])
    papers.append(dict(id=aid,title=re.sub(r"\s+"," ",e.find("a:title",ns).text).strip(),date=e.find("a:published",ns).text[:10],
      authors=[a.find("a:name",ns).text for a in e.findall("a:author",ns)][:4],
      abstract=re.sub(r"\s+"," ",e.find("a:summary",ns).text).strip()[:300],
      cat=e.find("{http://arxiv.org/schemas/atom}primary_category").get("term"),
      comment=(e.find("{http://arxiv.org/schemas/atom}comment").text if e.find("{http://arxiv.org/schemas/atom}comment") is not None else "")))
hf={}
for day in range(0,4):
    d=(dt.date.today()-dt.timedelta(days=day)).isoformat()
    try:
        for x in json.loads(G(f"https://huggingface.co/api/daily_papers?date={d}&limit=100")):
            p=x["paper"]; hf[p["id"]]=dict(up=p.get("upvotes",0),repo=p.get("githubRepo"),stars=p.get("githubStars"),proj=p.get("projectPage"))
    except Exception as ex: print("hf",d,ex)
for p in papers:
    h=hf.get(p["id"]); 
    if h: p["hf"]=h
    # code links announced by authors in arXiv comment/abstract
    p["code_in_text"]=re.findall(r"https?://github\.com/[\w.\-]+/[\w.\-]+",p["comment"]+" "+p["abstract"])
json.dump(dict(generated=str(dt.datetime.now(dt.timezone.utc))[:16]+"Z",papers=papers),open(os.path.join(HERE,"fresh.json"),"w"),indent=1,ensure_ascii=False)
print(len(papers),"arXiv papers, newest",papers[0]["date"] if papers else None,"| with HF signal:",sum('hf' in p for p in papers),"| with repo from HF:",sum(bool(p.get('hf',{}).get('repo')) for p in papers))
