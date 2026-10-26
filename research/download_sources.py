from pathlib import Path
from urllib.request import urlopen, Request
import concurrent.futures, json, hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
sources={
 'pi0.pdf':'https://www.pi.website/download/pi0.pdf',
 'pi05.pdf':'https://www.pi.website/download/pi05.pdf',
 'ki.pdf':'https://www.pi.website/download/pi05_KI.pdf',
 'pistar06.pdf':'https://www.pi.website/download/pistar06.pdf',
 'mem.pdf':'https://www.pi.website/download/Mem.pdf',
 'rlt.pdf':'https://www.pi.website/download/RLT.pdf',
 'pi07.pdf':'https://www.pi.website/download/pi07.pdf',
 'fast.pdf':'https://arxiv.org/pdf/2501.09747',
 'rtc.pdf':'https://arxiv.org/pdf/2506.07339',
 'rtc_training.pdf':'https://arxiv.org/pdf/2512.05964',
 'openpi_README.md':'https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/README.md',
 'openpi_pi0.py':'https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py',
 'openpi_pi0_config.py':'https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0_config.py',
 'openpi_config.py':'https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py',
 'openpi_train.py':'https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/scripts/train.py',
 'openpi_transforms.py':'https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py',
 'openpi_commit.json':'https://api.github.com/repos/Physical-Intelligence/openpi/commits/main',
}
def fetch(item):
 name,url=item
 try:
  data=urlopen(Request(url,headers={'User-Agent':'VLA-study-materials/1.0'}),timeout=90).read()
  (ROOT/name).write_bytes(data)
  if name.endswith('.pdf'):
   reader=PdfReader(ROOT/name)
   (ROOT/(name+'.txt')).write_text('\n\n'.join(f'=== PAGE {i+1} ===\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages)),encoding='utf-8')
  return {'file':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 except Exception as e:return {'file':name,'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 result=list(pool.map(fetch,sources.items()))
(ROOT/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
