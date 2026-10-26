from pathlib import Path
import re, math, html, json, shutil, zipfile
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, CondPageBreak, Image as RLImage, Table, TableStyle, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents

ROOT=Path(__file__).parent.resolve()
OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc'; BOLD='C:/Windows/Fonts/msyhbd.ttc'
W=1500; H=650
INK='#173245'; MUTED='#526878'; BLUE='#166d9c'; TEAL='#007f7a'; ORANGE='#bc681e'; RED='#b94c53'; BG='#f4f8fa'; LINE='#bdcdd7'
def fnt(n=34,b=False): return ImageFont.truetype(BOLD if b else FONT,n)
def wrap(text,width,font):
    result=[]
    for raw in str(text).replace('ᵀ','^T').replace('ᶜ','^C').split('\n'):
        line=''
        for c in raw:
            if font.getlength(line+c)>width and line:
                result.append(line); line=c
            else: line+=c
        result.append(line)
    return result
def canvas(title,subtitle='',h=650):
    im=Image.new('RGB',(W,h),'white'); d=ImageDraw.Draw(im)
    d.rectangle((0,0,12,76),fill=TEAL)
    d.text((40,12),title,font=fnt(40,True),fill=INK)
    if subtitle:d.text((42,72),subtitle,font=fnt(27),fill=MUTED)
    return im,d
def txt(d,xy,text,size=32,color=INK,b=False,width=None):
    text=str(text).replace('ᵀ','^T').replace('ᶜ','^C')
    font=fnt(size,b); lines=wrap(text,width,font) if width else str(text).split('\n')
    x,y=xy
    for line in lines:d.text((x,y),line,font=font,fill=color); y+=size*1.43
def box(d,x,y,w,h,title,body='',color=BLUE):
    d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=BG,outline=LINE,width=2)
    d.rectangle((x,y+15,x+5,y+h-15),fill=color)
    ts,bs=33,29
    while True:
        tl=wrap(title,w-36,fnt(ts,True)); bl=wrap(body,w-36,fnt(bs)) if body else []
        need=28+len(tl)*ts*1.35+len(bl)*bs*1.35
        if need<=h:break
        ts-=1;bs-=1
        if bs<24:raise ValueError(f'box overflow: {title}, {body}')
    yy=y+13
    for l in tl:d.text((x+18,yy),l,font=fnt(ts,True),fill=color); yy+=ts*1.35
    for l in bl:d.text((x+18,yy+5),l,font=fnt(bs),fill=INK); yy+=bs*1.35
def arrow(d,a,b,color=TEAL,width=5):
    d.line([a,b],fill=color,width=width)
    ang=math.atan2(b[1]-a[1],b[0]-a[0]); sz=15
    pts=[b,(b[0]-sz*math.cos(ang-.5),b[1]-sz*math.sin(ang-.5)),(b[0]-sz*math.cos(ang+.5),b[1]-sz*math.sin(ang+.5))]
    d.polygon(pts,fill=color)
def save(im,name): im.save(FIG/name)

im,d=canvas('从观测到经验改进：VLA 系统地图','箭头表示信息或反馈流；不同方法作用于不同环节')
box(d,45,145,350,170,'输入条件','图像、指令、本体状态\nMEM：补充历史信息')
box(d,570,145,350,170,'VLA 策略','Transformer + 动作生成\nFAST / Flow：输出表示')
box(d,1100,145,350,170,'实际执行','RTC：计算与执行衔接\n低层控制器跟踪目标')
arrow(d,(395,230),(570,230));arrow(d,(920,230),(1100,230))
box(d,570,425,350,155,'经验反馈','结果、失败、人类纠正\nRECAP / RLT：改进策略',TEAL)
arrow(d,(1280,315),(1280,505));arrow(d,(1280,505),(920,505));arrow(d,(745,425),(745,315))
txt(d,(50,415),'环境随动作改变\n下一帧观测也改变',34,width=400)
save(im,'01_system.png')

im,d=canvas('Attention：按相关性汇聚信息','先匹配 Q 与 K，再用权重汇聚 V；mask 限定哪些信息可见')
for x,t,b in [(45,'Q：查询','动作位置需要什么信息？'),(535,'K：索引','图像 / 指令如何匹配？'),(1025,'V：内容','最终要传递的特征')]: box(d,x,135,430,140,t,b)
box(d,290,360,420,175,'匹配与归一化','QKᵀ / sqrt(d_k) + mask\n逐行 softmax → 权重')
box(d,920,360,420,175,'加权汇聚','权重 × V\n得到新的 token 表示',TEAL)
arrow(d,(260,275),(430,360));arrow(d,(750,275),(580,360));arrow(d,(1240,275),(1140,360));arrow(d,(710,450),(920,450))
save(im,'02_attention.png')

im,d=canvas('B、K、H、d_a：四个独立维度','教学示例：B=8，K=6，H=50，d_a=14，三相机')
box(d,45,145,420,190,'B：样本数','8 个轨迹锚点\n用于估计一次更新的梯度')
box(d,540,145,420,190,'K：过去帧数','每个样本看 6 个历史时刻\n保留过去的信息')
box(d,1035,145,420,190,'H × d_a：未来标签','50 步 × 每步 14 维动作\n描述将来的控制目标')
txt(d,(70,400),'图像： [8, 6, 3, 高, 宽, 3]',43,b=True)
txt(d,(70,475),'动作： [8, 50, 14]    =    8 个样本 / 400 个时间位置',38,color=TEAL)
txt(d,(70,550),'同一 chunk 内的动作高度相关，不能按独立样本计数。',32,color=MUTED)
save(im,'03_tensor.png')

im,d=canvas('多峰策略与流匹配：生成一条自洽动作轨迹','左：两种有效路线的平均可能无效；右：生成时间不是真实控制时间')
d.line((95,530,640,530),fill=LINE,width=3);d.line((95,530,95,150),fill=LINE,width=3)
txt(d,(100,555),'空间 x',28);txt(d,(90,108),'空间 y',28)
d.rounded_rectangle((300,290,425,425),radius=8,fill='#dce4e9')
txt(d,(310,300),'障碍',30)
for yy,col in [(220,BLUE),(490,TEAL)]:
    pts=[(135,360),(240,yy),(470,yy),(600,360)];d.line(pts,fill=col,width=7)
d.line((135,360,600,360),fill=RED,width=4)
txt(d,(160,165),'路线 A',30,color=BLUE);txt(d,(440,555),'路线 B',30,color=TEAL)
box(d,780,160,580,150,'训练：学会向哪个方向变化','Xτ = (1−τ)ε + τA\n速度目标：A−ε')
box(d,780,390,580,150,'推理：从噪声逐步积分','噪声 → 中间动作 → 完整动作块\nτ：0 → 1；真实机器人尚未执行',TEAL)
arrow(d,(1070,310),(1070,390));save(im,'04_flow.png')

im,d=canvas('模型演进不是单一能力轴','上：基础模型与动作表示；下：实时、记忆和经验学习的配套方法',h=730)
nodes=[('π0','2024-10\n连续动作专家'),('π0-FAST','2025-01/02\n动作压缩 token'),('π0.5','2025-04\n新环境泛化'),('π0.6 / π*0.6','2025-11\n更大骨干 + RL'),('π0.7','2026-04\n多模态条件控制')]
for i,(t,b) in enumerate(nodes):
    x=30+i*295;box(d,x,145,265,215,t,b)
    if i<4:arrow(d,(x+265,245),(x+295,245))
for i,(t,b) in enumerate([('KI','隔离动作梯度\n2025-05'),('RTC','推理时 2025-06\n训练时 2025-12'),('MEM','短期 + 长期\n2026-03'),('RECAP / RLT','两条经验改进路线\n2025-11 / 2026-03')]):box(d,35+i*370,445,340,190,t,b,TEAL)
save(im,'05_evolution.png')

im,d=canvas('π0：大骨干理解条件，小专家生成连续动作','概念图；交互发生在 Transformer attention 层，而非只有末端线性头')
box(d,45,145,360,160,'图像 + 语言','视觉编码 / 分词 / 嵌入')
box(d,570,145,420,160,'VLM 骨干约 3B','条件表征；可缓存前缀 K/V')
box(d,45,420,360,160,'本体状态 + 带噪动作','投影并加入生成时间条件')
box(d,570,420,420,160,'动作专家约 300M','反复更新向量场\n动作 token 内部双向 attention',TEAL)
box(d,1130,420,320,160,'连续动作块','H × d_a\n反归一化后执行',TEAL)
arrow(d,(405,225),(570,225));arrow(d,(405,500),(570,500));arrow(d,(780,305),(780,420));arrow(d,(990,500),(1130,500))
save(im,'06_pi0.png')

im,d=canvas('FAST：先压缩时间结构，再学习离散 token','DCT 沿时间轴；量化有损；BPE 在固定词表下可逆')
for x,t,b in [(40,'① 归一化动作','每维统一尺度\n输入 H × d_a'),(540,'② DCT','时间序列 → 频率系数\n平稳变化集中在低频'),(1040,'③ 量化','系数 → 整数\n引入重建误差')]:box(d,x,145,420,170,t,b)
arrow(d,(460,230),(540,230));arrow(d,(960,230),(1040,230))
box(d,1040,420,420,150,'④ 展平','按固定顺序组织系数',TEAL);box(d,540,420,420,150,'⑤ BPE','常见序列合并为动作 token',TEAL);box(d,40,420,420,150,'自回归策略','训练预测 token；推理后逆解码',TEAL)
arrow(d,(1250,315),(1250,420));arrow(d,(1040,495),(960,495));arrow(d,(540,495),(460,495))
save(im,'07_fast.png')

im,d=canvas('高层子任务与低层动作：两种时间尺度','总任务指导子任务选择；最新观测持续修正低层执行')
box(d,45,160,380,165,'总任务','收拾桌面\n杯子进水槽，纸屑进垃圾桶')
box(d,565,160,380,165,'高层策略','当前子任务：\n把红杯子放进水槽',TEAL)
box(d,1085,160,370,165,'低层策略','生成关节与夹爪动作块\n连续控制目标')
arrow(d,(425,240),(565,240));arrow(d,(945,240),(1085,240))
box(d,565,450,380,120,'最新观测','确认抓取与任务进度',ORANGE)
arrow(d,(755,450),(755,325));arrow(d,(945,510),(1270,510));arrow(d,(1270,510),(1270,325))
save(im,'08_hierarchy.png')

im,d=canvas('Knowledge Insulation：前向能读，反向有边界','绿色：前向特征；红色：损失对参数的更新路径')
box(d,50,180,380,180,'VLM 骨干 θ','视觉、语言、状态\n仍接受离散目标训练')
box(d,590,135,370,165,'离散动作 / 文字头','CE：表征学习\n包含 FAST 与 VLM 任务')
box(d,590,420,370,165,'连续动作专家 φ','读取骨干特征\nFlow：连续运动生成',TEAL)
box(d,1110,420,340,165,'Flow loss','更新动作专家\n不沿此路径更新骨干',RED)
arrow(d,(430,225),(590,215));arrow(d,(430,330),(590,480));arrow(d,(960,500),(1110,500))
arrow(d,(595,145),(430,145),RED);arrow(d,(430,145),(250,180),RED)
txt(d,(438,370),'sg / 停梯度',27,color=RED);d.line((495,410,535,450),fill=RED,width=6);d.line((535,410,495,450),fill=RED,width=6)
txt(d,(45,540),'两种动作分支不能互读真实答案。',31,color=MUTED)
save(im,'09_ki.png')

im,d=canvas('π0.7：用丰富上下文指定行为','模型读取实际观测，也可使用语言、metadata 与子目标图像')
for i,(t,b) in enumerate([('任务与子任务','希望完成什么'),('近期视频与状态','刚才和现在发生了什么'),('行为质量 / 速度','metadata：行为条件'),('子目标图像','希望达到怎样的视觉状态')]):box(d,35+i*370,145,340,165,t,b)
box(d,420,435,650,150,'上下文条件化 VLA','MEM 风格历史编码 + 860M 动作专家\n通过训练时 RTC 生成可衔接动作块',TEAL)
for x in [205,575,945,1315]:arrow(d,(x,310),(745,435))
save(im,'10_pi07.png')

im,d=canvas('从停止等待到边执行边计算','教学时间轴：蓝色为执行，橙色为计算，绿色为约束后的新动作')
left=300;scale=15
for y,label in [(190,'同步'),(335,'朴素异步'),(480,'RTC')]:txt(d,(45,y+15),label,36,b=True)
def bar(x,y,w,col,label):
    d.rounded_rectangle((x,y,x+w,y+60),radius=8,fill=col)
    d.text((x+12,y+10),label,font=fnt(28),fill='white')
bar(left,190,750,BLUE,'旧块执行');bar(left+750,190,150,ORANGE,'等待推理');bar(left+900,190,200,TEAL,'新块执行')
bar(left,335,450,BLUE,'旧块继续');bar(left+450,335,650,RED,'独立新块：可能跳变')
bar(left+300,275,150,ORANGE,'计算')
bar(left,480,450,BLUE,'旧块继续');bar(left+450,480,650,TEAL,'新块：与承诺前缀相容')
bar(left+300,420,150,ORANGE,'计算')
txt(d,(left+280,570),'t=20 请求',28);txt(d,(left+470,570),'t=30 返回，跳过新块前 10 步',28)
save(im,'11_rtc.png')

im,d=canvas('RTC soft mask：越近的已承诺动作，约束越强','教学配置 H=50、s=20、d=10；曲线按论文权重公式计算')
x0=140;y0=535;pw=1220;ph=320
for v in [0,.5,1]:
    y=y0-v*ph;d.line((x0,y,x0+pw,y),fill=LINE,width=2);txt(d,(65,y-20),str(v),30)
for v in [0,10,20,30,40,49]:
    x=x0+v/49*pw;d.line((x,y0,x,y0+8),fill=INK,width=2);txt(d,(x-15,y0+16),str(v),28)
pts=[]
for i in range(50):
    c=(30-i)/21;v=1 if i<10 else c*(math.exp(c)-1)/(math.e-1) if i<30 else 0
    pts.append((x0+i/49*pw,y0-v*ph))
d.line(pts,fill=TEAL,width=7)
txt(d,(120,135),'约束权重 W',30,b=True);txt(d,(1040,590),'新 chunk 局部索引 i',28)
txt(d,(200,250),'已承诺\n0–9',30,color=BLUE);txt(d,(610,340),'重叠未承诺\n10–29：逐渐放松',30,color=TEAL);txt(d,(1070,435),'无旧计划\n30–49',30,color=MUTED)
save(im,'12_rtc_mask.png')

im,d=canvas('MEM：短期看细节，长期记事件','两条记忆通路有不同表示与更新节奏')
box(d,45,145,400,170,'短期：近期视频 + 状态','遮挡、运动、刚才的失败\n高信息密度，时间跨度短')
box(d,550,145,400,170,'时空视频编码器','空间 / 因果时间 attention\n历史信息压入当前表示')
box(d,1060,145,390,170,'低层 VLA','近期历史 + 子任务\n生成连续动作块',TEAL)
arrow(d,(445,230),(550,230));arrow(d,(950,230),(1060,230))
box(d,45,425,400,170,'长期：文本记忆','已完成步骤、对象位置\n压缩，不是无限追加日志')
box(d,550,425,400,170,'高层策略','读取当前观测与旧记忆\n输出子任务 + 新记忆',TEAL)
arrow(d,(445,510),(550,510));arrow(d,(950,510),(1255,510));arrow(d,(1255,510),(1255,315))
save(im,'13_mem.png')

im,d=canvas('多速率闭环：传感、高层、VLA 与控制器','示意节奏，无特定硬件性能含义；箭头强调数据版本与时间戳')
for y,label,col,step in [(160,'传感器',BLUE,75),(265,'高层 / 记忆',ORANGE,400),(370,'VLA 生成块',TEAL,220),(475,'低层控制器',BLUE,32)]:
    txt(d,(40,y+5),label,33,b=True);d.line((330,y+25,1420,y+25),fill=LINE,width=2)
    for x in range(350,1420,step): d.rounded_rectangle((x,y,x+22,y+50),radius=4,fill=col)
txt(d,(360,570),'每次动作块都应知道：观测时间、子任务版本、记忆版本。',32,color=MUTED)
save(im,'14_multirate.png')

im,d=canvas('RECAP：价值评价把混合经验变成策略改进信号','保留示范、失败与纠正；动作质量由优势条件区分')
box(d,45,150,400,175,'① 收集与累积','示范 + 自主执行 + 纠正\n记录真实结果与成败')
box(d,550,150,400,175,'② 学习价值 V','经验回报 → 价值分布\n比较实际进度与预期')
box(d,1060,150,390,175,'③ 生成优势标签','A_hat 与任务阈值比较\n得到正 / 负优势条件')
arrow(d,(445,235),(550,235));arrow(d,(950,235),(1060,235))
box(d,550,430,400,175,'④ 训练条件 VLA','生成目标仍可用 CE / Flow\n部署偏向正优势条件',TEAL)
arrow(d,(1255,325),(1255,515));arrow(d,(1255,515),(950,515));arrow(d,(550,515),(245,515));arrow(d,(245,515),(245,325))
save(im,'15_recap.png')

im,d=canvas('RLT：保留大模型先验，在线优化小网络','适配完成后，VLA 与 RL token 模块冻结；actor / critic 在线更新')
box(d,45,150,400,170,'冻结 VLA','内部特征 → RL token\n动作专家 → 参考动作块')
box(d,560,150,400,170,'轻量 actor','输入 z_rl、状态、参考块\n输出改进动作块',TEAL)
box(d,1070,150,380,170,'机器人 / replay','实际动作、奖励、后继状态\n也可包含人类纠正')
arrow(d,(445,235),(560,235));arrow(d,(960,235),(1070,235))
box(d,560,430,400,170,'轻量 critic','chunk 级 TD 评价\nactor 追求高 Q，约束偏离',ORANGE)
arrow(d,(1260,320),(1260,515));arrow(d,(1260,515),(960,515));arrow(d,(760,430),(760,320),ORANGE)
txt(d,(45,445),'RL token 是连续特征，\n不是文字奖励标签。',31,width=440)
save(im,'16_rlt.png')

im,d=canvas('验证分三层：任何一层出错，都可能导致失败','评测采用事先定义的任务、数据划分与初始状态')
box(d,40,155,440,250,'数据层','时间对齐 / 单位 / 坐标系\nepisode 与场景分割\n归一化与动作逆变换\n缺失相机和 padding')
box(d,530,155,440,250,'策略层','离线损失与动作诊断\n成功率与失败原因\n恢复能力与语言服从\n未见对象 / 场景评测',TEAL)
box(d,1020,155,440,250,'系统层','端到端 P95 / P99 延迟\n队列耗尽与块切换\n真实墙钟吞吐\n人工介入与重置成本',ORANGE)
txt(d,(75,480),'训练 loss ↓  ≠  真实任务成功率必然 ↑',42,b=True,color=TEAL)
txt(d,(75,555),'同一套测试协议，才能比较模型、RTC、记忆和 RL 的独立贡献。',32)
save(im,'17_eval.png')

im,d=canvas('整理厨房：同一个任务中的方法分工','任务：杯子入柜 → 擦干台面 → 关闭柜门')
for i,(t,b,col) in enumerate([('理解与动作','VLM 识别杯子与指令\nFlow 生成抓取动作',BLUE),('执行与记忆','RTC 保持块切换连贯\nMEM 记住柜门未关',TEAL),('经验与改进','RECAP 改善重复失败\nRLT 精调关键接触',ORANGE)]):box(d,40+i*495,165,440,220,t,b,col)
arrow(d,(480,270),(535,270));arrow(d,(975,270),(1030,270))
txt(d,(75,465),'实际执行结果 → 更新记忆、形成经验 → 下一次决策',38,b=True,color=TEAL)
txt(d,(75,545),'计划、文字摘要、目标图像都需要与真实物理结果核对。',32,color=MUTED)
save(im,'18_case.png')

# Combine editable source and create a self-contained offline reading edition.
text='\n\n'.join(p.read_text(encoding='utf-8') for p in sorted((ROOT/'materials').glob('*.md'))).replace('ᵀ','^T').replace('ᶜ','^C')
(OUT/'VLA_详细图解讲义.md').write_text(text,encoding='utf-8')
pdfmetrics.registerFont(TTFont('Yahei',FONT));pdfmetrics.registerFont(TTFont('YaheiBold',BOLD))
pdfmetrics.registerFontFamily('Yahei',normal='Yahei',bold='YaheiBold',italic='Yahei',boldItalic='YaheiBold')
styles={
'body':ParagraphStyle('body',fontName='Yahei',fontSize=10.2,leading=16.2,spaceAfter=7,wordWrap='CJK',textColor=colors.HexColor(INK)),
'h2':ParagraphStyle('h2',fontName='YaheiBold',fontSize=21,leading=30,spaceAfter=17,wordWrap='CJK',textColor=colors.HexColor(INK),keepWithNext=True),
'h3':ParagraphStyle('h3',fontName='YaheiBold',fontSize=13,leading=21,spaceBefore=12,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor(TEAL),keepWithNext=True),
'caption':ParagraphStyle('caption',fontName='Yahei',fontSize=8.4,leading=13,spaceAfter=13,textColor=colors.HexColor(MUTED),wordWrap='CJK'),
'quote':ParagraphStyle('quote',fontName='Yahei',fontSize=10.2,leading=18,backColor=colors.HexColor('#eef5f7'),borderPadding=9,spaceBefore=8,spaceAfter=14,wordWrap='CJK'),
'cell':ParagraphStyle('cell',fontName='Yahei',fontSize=8.8,leading=14,wordWrap='CJK'),
'code':ParagraphStyle('code',fontName='Yahei',fontSize=8.9,leading=14,wordWrap='CJK',backColor=colors.HexColor('#f0f4f6'),borderPadding=7,spaceAfter=10),
}
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<a href="{m[2]}" color="#166d9c">{m[1]}</a>',s)
    s=re.sub(r'`([^`]+)`',r'<font color="#166d9c">\1</font>',s)
    return s
class Doc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=(595.28,841.89),leftMargin=46,rightMargin=46,topMargin=55,bottomMargin=49,title='VLA 详细图解讲义',author='技术学习资料 · Codex',pageCompression=1)
        self.addPageTemplates(PageTemplate('main',[Frame(46,49,503.28,737.89,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.decoration))
    def decoration(self,c,doc):
        if doc.page==1:return
        c.setStrokeColor(colors.HexColor('#d8e4e9'));c.line(46,808,549,808)
        c.setFont('Yahei',8);c.setFillColor(colors.HexColor(MUTED));c.drawString(46,818,'VLA / π 系列 / RTC / MEM / RL')
        c.drawRightString(549,29,f'{doc.page:02d}');c.drawString(46,29,'技术核验：2026-10-25 · 原始资料见末章')
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='h2' and re.match(r'^\d{2}',f.getPlainText()):
            t=f.getPlainText(); key='ch'+re.sub(r'[^0-9]','',t[:2]);self.canv.bookmarkPage(key);self.canv.addOutlineEntry(t,key,0)
            self.notify('TOCEntry',(0,t,self.page,key))

story=[]
story.append(Spacer(1,70))
for s,sz,col in [('具身智能 · 技术学习手册',13,TEAL),('VLA',58,INK),('从基础到 π 系列',31,INK),('实时控制、记忆与强化学习',21,TEAL)]:
    story.append(Paragraph(s,ParagraphStyle('cover',fontName='YaheiBold',fontSize=sz,leading=sz*1.5,textColor=colors.HexColor(col),spaceAfter=12)))
story.append(Spacer(1,20));story.append(RLImage(str(FIG/'01_system.png'),width=503,height=218))
story.append(Spacer(1,22));story.append(Paragraph('22 个学习单元 · 18 张原创示意图 · 19 项原始资料<br/>公式、张量形状、训练参数、代码导航与自测答案',styles['body']))
story.append(Paragraph('核验日期：2026 年 10 月 25 日<br/>论文事实 / 通用原理 / 教学示例 / 工程建议分开说明',styles['caption']))
story.append(PageBreak())
story.append(Paragraph('阅读目录',styles['h2']))
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='Yahei',fontSize=10.5,leading=20,leftIndent=0,firstLineIndent=0,textColor=colors.HexColor(INK),wordWrap='CJK')]
story.append(toc);story.append(PageBreak())
story.append(Paragraph('使用说明',styles['h2']))
lines=text.splitlines();i=1;html_parts=[];toc_html=[];chapter=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('## '):
        title=line[3:];chapter+=1;anchor='section-'+str(chapter)
        story.append(PageBreak() if chapter in [1,7,11,14,17,22] else CondPageBreak(245));story.append(Paragraph(inline(title),styles['h2']))
        html_parts.append(f'<h2 id="{anchor}">{html.escape(title)}</h2>');toc_html.append(f'<a href="#{anchor}">{html.escape(title)}</a>');i+=1
    elif line.startswith('### '):
        if line[4:]=='答案与解释':story.append(PageBreak())
        story.append(Paragraph(inline(line[4:]),styles['h3']));html_parts.append('<h3>'+inline(line[4:])+'</h3>');i+=1
    elif line.startswith('!['):
        m=re.match(r'!\[([^\]]+)\]\(([^)]+)\)',line);p=OUT/m[2];pic=Image.open(p);w=503;h=w*pic.height/pic.width
        story.append(KeepTogether([Spacer(1,6),RLImage(str(p),width=w,height=h),Spacer(1,5),Paragraph(inline(m[1])+' · 原创教学示意',styles['caption'])]))
        import base64
        data=base64.b64encode(p.read_bytes()).decode();html_parts.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(m[1])}"/><figcaption>{html.escape(m[1])} · 原创教学示意</figcaption></figure>');i+=1
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',x) for x in row):rows.append(row)
            i+=1
        cols=len(rows[0]); widths=[503/cols]*cols
        if cols==3:widths=[110,184,209]
        if cols==4:widths=[100,135,130,138]
        cells=[[Paragraph(inline(x),styles['cell']) for x in row] for row in rows]
        table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f1f4')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafb')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor(LINE)),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#e4ebef'))]))
        story.extend([table,Spacer(1,12)])
        html_parts.append('<div class="tablewrap"><table>'+''.join('<tr>'+''.join(f'<{"th" if n==0 else "td"}>{inline(x)}</{"th" if n==0 else "td"}>' for x in row)+'</tr>' for n,row in enumerate(rows))+'</table></div>')
    elif line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        i+=1
        # Paragraph handles Chinese line wrapping; preserve indent with non-breaking spaces.
        story.append(Paragraph('<br/>'.join(html.escape(x).replace('  ','&#160;&#160;') for x in code),styles['code']))
        html_parts.append('<pre>'+html.escape('\n'.join(code))+'</pre>')
    elif line.startswith('> '):
        q=[]
        while i<len(lines) and lines[i].startswith('> '):q.append(lines[i][2:]);i+=1
        story.append(Paragraph('<br/>'.join(inline(x) for x in q),styles['quote']));html_parts.append('<blockquote>'+'<br/>'.join(inline(x) for x in q)+'</blockquote>')
    elif re.match(r'^\d+\. ',line):
        story.append(Paragraph(inline(line),styles['body']));html_parts.append('<p>'+inline(line)+'</p>');i+=1
    else:
        p=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|```|> |\d+\. )',lines[i]):p.append(lines[i].strip());i+=1
        para=' '.join(p);story.append(Paragraph(inline(para),styles['body']));html_parts.append('<p>'+inline(para)+'</p>')

pdf=OUT/'VLA_详细图解讲义.pdf';doc=Doc(pdf);doc.multiBuild(story)

calc='''<section class="calc"><h2>交互算例：把四种“步数”分开</h2><p>以下均为教学参数，不是 PI 的推荐训练配置。改变数字可观察维度关系。</p>
<div class="calcgrid"><div><h3>有效 batch 与训练预算</h3><label>每副本 micro batch <input id="b" type="number" min="1" value="4"></label><label>数据并行副本 D <input id="dp" type="number" min="1" value="4"></label><label>梯度累计 G <input id="g" type="number" min="1" value="8"></label><label>固定窗口总数 N <input id="n" type="number" min="1" value="96000"></label><output id="batchout" aria-live="polite"></output></div>
<div><h3>RTC 等待覆盖</h3><label>控制频率 Hz <input id="hz" type="number" min="1" value="50"></label><label>预测 H <input id="h" type="number" min="1" value="50"></label><label>请求启动前已前进 s <input id="s" type="number" min="0" value="20"></label><label>端到端延迟 ms <input id="lat" type="number" min="0" value="200"></label><output id="rtcout" aria-live="polite"></output></div></div>
<p class="small">B_eff=b×D×G 假设纯数据并行且正确归一化；RTC 采用保守向上取整。此算例不模拟真实机器人动力学。</p></section>'''
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLA 详细图解讲义</title><style>
:root{color-scheme:light;--ink:#173245;--teal:#007f7a}*{box-sizing:border-box}body{margin:0;color:var(--ink);font:17px/1.9 "Microsoft YaHei",system-ui,sans-serif;background:#f5f8fa}header{background:#153749;color:white;padding:48px max(24px,calc((100vw - 1020px)/2));}header h1{font-size:40px;line-height:1.4;margin:0 0 16px}header p{margin:0;color:#d3e7ef}main{max-width:1080px;margin:auto;background:white;padding:40px 58px}h2{font-size:27px;line-height:1.5;margin-top:64px;padding-top:22px;border-top:3px solid #e4edef;scroll-margin-top:20px}h3{color:var(--teal);font-size:21px;margin-top:32px}p{margin:15px 0}nav{columns:2;column-gap:35px}nav a{display:block;break-inside:avoid;font-size:16px;padding:5px 0}a{color:#166d9c}figure{margin:28px 0}img{width:100%;height:auto}figcaption,.small{font-size:14px;color:#526878}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.65}td,th{padding:11px 13px;border-bottom:1px solid #d9e3e9;text-align:left;vertical-align:top}th{background:#e7f1f4}tr:nth-child(even){background:#f8fafb}.tablewrap{overflow:auto}blockquote{background:#eef5f7;border-left:4px solid var(--teal);margin:22px 0;padding:16px 20px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.8 "Microsoft YaHei",sans-serif;background:#f0f4f6;padding:20px}.calc{margin:32px 0;padding:24px;background:#f4f8fa}.calc h2{margin-top:0;border:0;padding:0}.calcgrid{display:grid;grid-template-columns:1fr 1fr;gap:35px}label{display:flex;justify-content:space-between;gap:20px;margin:10px 0;font-size:15px}input{width:120px;padding:6px 9px;border:1px solid #b5c7d1;border-radius:5px;font:inherit}output{display:block;background:#e1f0ed;padding:12px;font-weight:bold;margin-top:15px}button{font:inherit;padding:8px 16px;border:1px solid #abc3cf;background:white;border-radius:5px;color:var(--ink);cursor:pointer}.tools{display:flex;gap:12px;margin-bottom:25px}::selection{background:#bbdfd6}@media(max-width:700px){main{padding:25px 20px}header{padding:30px 20px}header h1{font-size:30px}nav{columns:1}.calcgrid{grid-template-columns:1fr}h2{font-size:24px}td,th{min-width:110px}}@media print{body{background:white}header{color:#173245;background:white;padding:20px}.calc,.tools{display:none}main{padding:0;max-width:none}h2{break-before:page}figure,table{break-inside:avoid}a{color:inherit}}
</style></head><body><header><h1>VLA：从基础到 π 系列</h1><p>Transformer · 训练与微调 · RTC · MEM · RECAP · RLT</p><p>详细图解讲义 / 核验日期 2026-10-25 / 离线阅读版</p></header><main><div class="tools"><button onclick="window.print()">打印</button><button onclick="window.scrollTo({top:0,behavior:'smooth'})">回到顶部</button></div><p>浏览器内可用 Ctrl+F 全文检索；图片已嵌入，离线也能阅读。点击目录跳转，文末链接指向原始资料。</p><nav>'''+''.join(toc_html)+'</nav>'+calc+''.join(html_parts)+'''</main><script>
function update(){let v=id=>Number(document.getElementById(id).value);let b=v('b'),D=v('dp'),G=v('g'),N=v('n'),f=v('hz'),H=v('h'),s=v('s'),lat=v('lat');let B=b*D*G;document.getElementById('batchout').textContent=[b,D,G,N].every(x=>Number.isInteger(x)&&x>0)?`有效 batch = ${B} 个窗口；每轮约 ${(N/B).toFixed(2)} 次更新（忽略丢尾）。`:'请填写正整数。';let d=Math.ceil(lat*f/1000),remain=H-s;document.getElementById('rtcout').textContent=f>0&&Number.isInteger(H)&&Number.isInteger(s)&&s>=0&&s<=H&&lat>=0?`延迟约 ${d} 个控制步；旧块剩余 ${remain} 步；${d<=remain?'等待覆盖条件满足':'旧块不够覆盖等待'}。预测跨度 ${(H/f).toFixed(2)} 秒。`:'请检查输入：0 ≤ s ≤ H，H、频率须为正。'}document.querySelectorAll('input').forEach(e=>e.addEventListener('input',update));update();
</script></body></html>'''
(OUT/'VLA_详细图解讲义.html').write_text(page,encoding='utf-8')
summary={'characters':len(text),'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',text)),'figures':len(list(FIG.glob('*.png'))),'chapters':chapter,'sources':19}
(OUT/'build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary))
