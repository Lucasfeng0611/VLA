from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader
import json,re
ROOT=Path(__file__).parent; out=ROOT/'output'; qa=ROOT/'tmp'/'pdfs';qa.mkdir(parents=True,exist_ok=True)
p=pdfium.PdfDocument(str(out/'VLA_详细图解讲义.pdf'))
thumbs=[]
for i in range(len(p)):
    img=p[i].render(scale=1.15).to_pil().convert('RGB');img.save(qa/f'page-{i+1:02d}.png')
    img.thumbnail((290,410));thumbs.append(img)
for start in range(0,len(thumbs),12):
    sheet=Image.new('RGB',(1200,1335),'#dde5e9');d=ImageDraw.Draw(sheet)
    for j,img in enumerate(thumbs[start:start+12]):
        x=(j%4)*300+5;y=(j//4)*445+22;sheet.paste(img,(x,y));d.text((x,y-18),f'Page {start+j+1}',fill='black')
    sheet.save(qa/f'contact-{start//12+1}.png')
figs=list((out/'figures').glob('*.png'))
for start in range(0,len(figs),6):
    sheet=Image.new('RGB',(1500,1100),'#dde5e9')
    for j,f in enumerate(figs[start:start+6]):
        img=Image.open(f);img.thumbnail((735,335));sheet.paste(img,((j%2)*750+5,(j//2)*365+5))
    sheet.save(qa/f'figures-{start//6+1}.png')
r=PdfReader(str(out/'VLA_详细图解讲义.pdf'))
text='\n'.join(x.extract_text() for x in r.pages)
checks={'pages':len(r.pages),'rendered_pages':len(thumbs),'figures':len(figs),'links':sum(len(x.get('/Annots',[])) for x in r.pages),'toc_present':'阅读目录' in text,'chapters_found':len(set(re.findall(r'\b\d{2}｜',text))),'replacement_glyph':'\ufffd' in text}
(qa/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=True))
