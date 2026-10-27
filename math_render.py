"""Offline TeX math: SVG for HTML, 600 dpi assets for PDF, no CDN required."""
from pathlib import Path
import base64, hashlib, html, json, os, sys

ROOT=Path(__file__).parent.resolve()
sys.path.insert(0,str(ROOT/'.dependencies'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'mathtext.fontset':'stix','svg.fonttype':'path'})
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from matplotlib.figure import Figure

DIR=ROOT/'output'/'math';DIR.mkdir(parents=True,exist_ok=True)
PARSER=MathTextParser('path')
FONT=FontProperties(size=12)
CACHE={}

def asset(tex):
    if tex in CACHE:return CACHE[tex]
    key=hashlib.sha256(tex.encode()).hexdigest()[:16]
    formula='$'+tex+'$'
    width,height,depth,*_=PARSER.parse(formula,dpi=72,prop=FONT)
    pad=1.0; w=float(width)+2*pad;h=float(height)+2*pad;dep=float(depth)+pad
    png=DIR/(key+'.png');svg=DIR/(key+'.svg')
    if not png.exists() or not svg.exists():
        fig=Figure(figsize=(w/72,h/72),dpi=600)
        fig.text(pad/w,dep/h,formula,fontproperties=FONT,color='#173245')
        fig.savefig(png,dpi=600,transparent=True)
        fig.savefig(svg,format='svg',transparent=True)
        fig.clear()
    data={'tex':tex,'width':w,'height':h,'depth':dep,'png':str(png),'svg':str(svg)}
    CACHE[tex]=data
    return data

def inline_math(tex,target='pdf',size=10.2):
    a=asset(tex)
    if target=='html':
        data=base64.b64encode(Path(a['png']).read_bytes()).decode()
        return (f'<img class="math-inline" src="data:image/png;base64,{data}" '
                f'alt="{html.escape(tex,quote=True)}" style="width:{a["width"]/12:.4f}em;'
                f'height:{a["height"]/12:.4f}em;vertical-align:-{a["depth"]/12:.4f}em">')
    scale=size/12
    return f'<img src="{a["png"]}" width="{a["width"]*scale:.3f}" height="{a["height"]*scale:.3f}" valign="{-a["depth"]*scale:.3f}"/>'

def display_math(tex):
    from reportlab.platypus import Image,Spacer,KeepTogether
    a=asset(tex);scale=min(13/12,495/a['width'])
    im=Image(a['png'],width=a['width']*scale,height=a['height']*scale)
    im.hAlign='CENTER'
    data=base64.b64encode(Path(a['svg']).read_bytes()).decode()
    block=f'<div class="math-display">{inline_math(tex,"html")}</div>'
    return KeepTogether([Spacer(1,7),im,Spacer(1,12)]),block

def manifest():
    (DIR/'manifest.json').write_text(json.dumps(list(CACHE.values()),ensure_ascii=False,indent=2),encoding='utf-8')
