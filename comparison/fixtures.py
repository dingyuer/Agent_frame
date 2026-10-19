"""Deterministic, synthetic evidence fixtures. Expected labels never enter prompts."""
from functools import lru_cache
import base64
import hashlib
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .presentation import question_ui

ASSETS = Path(__file__).with_name('assets')
CATEGORIES = {'readiness':'等待与恢复', 'verification':'动作与完成验证', 'routing':'DOM / GUI 路由', 'selection':'目标与标签选择', 'typed':'结构化决策'}


def font(size, bold=False):
    paths = [f'/usr/share/fonts/truetype/dejavu/DejaVuSans{"-Bold" if bold else ""}.ttf',
             f'C:/Windows/Fonts/arial{"bd" if bold else ""}.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render(kind):
    im = Image.new('RGB', (1000, 640), '#f5f7fb')
    d = ImageDraw.Draw(im)
    def txt(x, y, text, size=18, color='#24324a', bold=False):
        d.text((x,y), text, fill=color, font=font(size,bold))
    def box(rect, fill='white', outline='#dce3ef', radius=10):
        d.rounded_rectangle(rect,radius=radius,fill=fill,outline=outline,width=2)
    def button(x,y,w,text,fill='#4263eb'):
        box((x,y,x+w,y+44),fill,fill,8); txt(x+18,y+10,text,16,'white',True)
    box((0,0,999,70),'#e8edf5','#e8edf5',0)
    for x,col in [(22,'#ff6d66'),(44,'#ffc25a'),(66,'#60c58a')]: d.ellipse((x,16,x+11,27),fill=col)
    txt(104,10,'Research workspace',14)
    box((110,37,936,63),'white','white',7)
    url='https://papers.example/search'
    if kind.startswith('paper'): url='https://papers.example/abs/2603.11558'
    if kind.startswith('pdf') or kind.startswith('download'): url='https://papers.example/pdf/2603.11558'
    if kind=='native': url='File Explorer / Save As'
    txt(129,40,url,13,'#607088')
    if kind in ('loading','skeleton'):
        if kind=='loading':
            d.rectangle((0,71,1000,640),fill='#111827')
            d.arc((466,260,534,328),10,290,fill='#8b9cf6',width=7)
            txt(404,355,'Loading document...',19,'#b4c1d9')
        else:
            txt(44,110,'Search results',27,bold=True)
            for y in (185,305,425):
                box((42,y,958,y+96),'white')
                for xx,yy,w in [(65,y+20,540),(65,y+53,790)]: box((xx,yy,xx+w,yy+18),'#e5eaf2','#e5eaf2',5)
            txt(44,573,'Fetching results...',16,'#8692a6')
    elif kind.startswith('search'):
        txt(44,108,'Paper Library',30,bold=True); txt(45,153,'Find research. Follow the evidence.',16,'#718096')
        box((44,204,766,260)); txt(64,220,'RoboClaw' if kind!='search_empty' else 'Search papers...',19,'#607088')
        button(784,208,170,'Search')
        if kind=='search_results':
            txt(44,302,'3 results for RoboClaw',18,bold=True)
            for i,title in enumerate(['RoboClaw: An Agentic Framework for Robotic Tasks','A Survey of Robotic Foundation Models','Efficient Claw Manipulation']):
                box((44,344+i*88,956,420+i*88)); txt(64,360+i*88,title,18,bold=True); txt(64,389+i*88,'Abstract    View PDF',14,'#4263eb')
        else:
            txt(44,323,'Browse recent research',20,bold=True)
            txt(44,365,'Enter a query and press Search to see results.',18,'#8793a6')
    elif kind.startswith('paper'):
        wrong=kind=='paper_wrong'
        txt(44,103,'Paper Library / Robotics',18,'#7b4669',True)
        title='A Survey of Robotic Foundation Models' if wrong else 'RoboClaw: An Agentic Framework'
        txt(44,166,title,29,bold=True)
        txt(44,210,'for Robotic Tasks' if not wrong else 'Related work mentioning RoboClaw',24)
        txt(44,266,'A. Example, B. Researcher  |  Synthetic demo paper',15,'#7c8798')
        txt(44,325,'Abstract',22,bold=True)
        for i,line in enumerate(['We study long-horizon robotic tasks and agent systems.', 'This page is a controlled fixture for comparing decisions.', 'It is not a captured session or a real research claim.']): txt(44,366+i*30,line,18,'#607088')
        button(44,510,170,'View PDF')
    elif kind.startswith('pdf') or kind.startswith('download'):
        d.rectangle((0,72,1000,132),fill='#344156'); txt(27,91,'2603.11558.pdf',19,'white'); txt(458,94,'1 / 18',16,'white')
        txt(780,94,'100%    Save    Print',16,'white')
        box((170,152,830,660),'white')
        txt(210,215,'RoboClaw: An Agentic Framework',25,bold=True); txt(210,253,'for Robotic Tasks',25,bold=True)
        txt(210,310,'A. Example and B. Researcher',16,'#718096')
        txt(210,370,'Abstract',20,bold=True)
        for y,w in [(418,542),(449,510),(480,535),(511,470),(542,525)]: box((210,y,210+w,y+9),'#d4dce7','#d4dce7',2)
        if kind.startswith('download'):
            box((650,146,980,300),'white')
            txt(670,164,'Downloads',18,bold=True); txt(670,204,'RoboClaw.pdf',17)
            if kind=='download_pending':
                txt(670,240,'3.2 / 8.0 MB  -  40%',15,'#4263eb'); box((670,273,952,280),'#e0e7ff','#e0e7ff'); box((670,273,783,280),'#4263eb','#4263eb')
            elif kind=='download_done': txt(670,246,'Completed  -  8.0 MB',16,'#0a9168')
            else: txt(670,246,'Network error - Retry',16,'#d64859')
    elif kind in ('overlay','form','form_saved','checkbox','focus_wrong'):
        txt(46,111,'Workspace settings',29,bold=True)
        txt(46,175,'Project name',17); box((46,208,620,260)); txt(65,224,'RoboClaw',19)
        txt(46,293,'Contact email',17); box((46,326,620,378),outline='#4263eb' if kind=='focus_wrong' else '#dce3ef'); txt(65,342,'RoboClaw' if kind=='focus_wrong' else 'team@example.org',18)
        box((47,414,70,437),radius=4); txt(85,413,'Enable automatic backups',17)
        button(46,490,160,'Save changes')
        txt(235,504,'All changes saved' if kind=='form_saved' else 'Unsaved changes',17,'#0a9168' if kind=='form_saved' else '#b06a21')
        if kind=='overlay':
            mask=Image.new('RGBA',im.size,(20,30,50,135)); im=Image.alpha_composite(im.convert('RGBA'),mask).convert('RGB'); d=ImageDraw.Draw(im)
            box((240,177,760,463),'white'); txt(275,212,'Session expired',28,bold=True)
            txt(275,275,'Sign in again to continue editing.',19,'#607088'); button(275,365,180,'Sign in again')
    elif kind=='canvas':
        txt(40,104,'Workflow canvas',28,bold=True)
        for x in range(30,990,22):
            for y in range(164,620,22): d.ellipse((x,y,x+2,y+2),fill='#d5dce8')
        for x,label in [(80,'Input'),(390,'Transform'),(700,'Export PDF')]:
            box((x,290,x+220,392),'white','#b9c5e5'); txt(x+24,326,label,20,bold=True)
        d.line((300,341,390,341),fill='#99a8c2',width=3); d.line((610,341,700,341),fill='#99a8c2',width=3)
        txt(45,568,'Nodes are drawn on a canvas surface.',17,'#718096')
    elif kind=='native':
        box((80,110,920,590),'white'); txt(108,136,'Save As',26,bold=True)
        box((106,193,894,236),'#f0f3f9'); txt(122,204,'This PC  >  Documents  >  Reports',17)
        txt(114,270,'Name                                      Type',17,'#718096')
        txt(114,316,'Previous report.pdf                      PDF document',18)
        txt(114,421,'File name',17); box((230,410,890,458)); txt(245,425,'RoboClaw.pdf',18)
        button(590,498,135,'Save'); button(746,498,140,'Cancel','#738099')
    elif kind.startswith('tab'):
        title={'tab_scholar':'Scholar search: RoboClaw','tab_github':'Repository / RoboClaw','tab_blank':'New tab','tab_unknown':'Loading...'}[kind]
        txt(50,143,title,32,bold=True)
        txt(50,210,'Search results and citations' if kind=='tab_scholar' else 'Source code, issues and pull requests' if kind=='tab_github' else 'No document content is available.',19,'#718096')
    else: raise ValueError(kind)
    return im


def asset(kind):
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / (kind+'.png')
    if not path.exists():
        render(kind).save(path, 'PNG', optimize=True)
    return kind


def choice(criteria, instructions):
    return {'type':'choice','criteria':criteria,'instructions':instructions}

READINESS=choice({'ready':'页面已渲染出可操作的内容。', 'waiting':'页面仍是空白、加载动画或骨架占位内容。', 'unknown':'现有证据不足以判断页面是否就绪。'}, '根据当前截图和 AX 信息判断。出现加载占位内容不代表页面已经就绪。')
REPLAN=choice({'yes':'当前策略受到阻碍，需要调整计划。', 'no':'当前计划仍可继续，或只需短暂等待。', 'unknown':'证据不足，暂时无法判断。'}, '短暂加载本身不构成重新规划的理由。')
ROUTE=choice({'dom':'当前目标有可用的 AX / DOM 引用，可以直接通过结构化接口操作。', 'gui':'目标位于原生窗口、浏览器工具栏或画布中，没有可用的 AX / DOM 引用，应使用视觉定位。', 'unknown':'尚不能确定具体目标，或无法确认目标是否可操作。'}, '围绕指定的下一步目标选择操作方式。页面存在其他 DOM 节点，并不代表当前目标也有可用引用。')
IDENTITY=choice({'yes':'当前可见标题与任务要求的完整论文标题一致。', 'no':'当前标题对应另一篇论文，并非任务要求的目标。', 'unknown':'现有证据不足以确认论文身份。'}, '核对目标论文的完整标题；仅提及相同关键词不能证明是同一篇论文。')
DONE=choice({'done':'同时存在下载完成事件与校验通过的已保存 PDF 文件。', 'wait':'文件仍在传输，应继续等待。', 'retry':'存在明确的下载失败证据，应恢复或重试。', 'inspect':'证据不足以确认下载完成，应先检查当前页面或文件。'}, '任务要求将有效 PDF 保存到磁盘。打开 PDF 预览或派发点击不代表文件已保存；请结合浏览器事件与文件校验证据判断。')
TARGET='RoboClaw: An Agentic Framework for Robotic Tasks'


def case(id,title,category,kinds,task,state,questions,expected,insight,*,featured=False,section='主案例',pair=None):
    ui = question_ui(questions)
    questions = {key: {**q, 'instructions': ui[key]['title'] + q['instructions']} for key, q in questions.items()}
    return dict(id=id,title=title,category=category,category_name=CATEGORIES[category],assets=[asset(k) for k in kinds],
                task=task,state={'task':task, **state},questions=questions,expected=expected,insight=insight,
                question_ui=ui,prompt_language='zh-CN',
                featured=featured,section=section,pair=pair,source='synthetic_fixture',
                source_label='受控合成场景 · 非真实任务轨迹')


def build_cases():
    out=[]
    def add(*a,**kw): out.append(case(*a,**kw))
    add('arxiv_dom_gui','正文链接：选择 DOM','routing',['paper_correct'],'打开以下目标论文的 PDF：'+TARGET,
        {'ax':'ref=17 role=link name="View PDF" href="/pdf/2603.11558"; ref=12 role=heading name="'+TARGET+'"','next_target':'“View PDF（查看 PDF）”链接'},
        {'route':ROUTE,'identity':IDENTITY},{'route':'dom','identity':'yes'},'同一调用同时检查目标身份与操作路由；精确 ref 是关键证据。',pair='canvas_gui')
    add('loading_black','黑屏加载：等待，不重规划','readiness',['loading'],'等待文档加载完成，直到页面可以操作。',
        {'ax':'','elapsed_since_navigation_ms':1200},{'readiness':READINESS,'replan':REPLAN},{'readiness':'waiting','replan':'no'},'高频等待轮询适合短决策。比较端到端耗时和结构，不预设谁更快。',featured=True,pair='search_results')
    add('loading_skeleton','骨架屏：有布局也未就绪','readiness',['skeleton'],'读取搜索结果中的第一条论文。',
        {'ax':'heading Search results; status Fetching results; 没有结果链接'}, {'readiness':READINESS},{'readiness':'waiting'},'黑屏之外，再检查看似完整的占位内容，避免点击尚未出现的结果。')
    add('overlay_blocked','遮罩拦截：先恢复会话','readiness',['overlay'],'保存已经编辑好的项目设置。',
        {'ax':'dialog Session expired; ref=91 button Sign in again; 背景中的 Save changes 按钮不可交互'},
        {'next':choice({'dismiss':'先处理前景对话框，重新登录并恢复会话。','save':'直接点击背景中的“Save changes（保存改动）”按钮。','wait':'继续等待页面加载。','unknown':'证据不足，暂时无法判断。'},'当前模态对话框会拦截背景控件，请据此决定下一步操作。')},{'next':'dismiss'},'页面不是空白，但目标不可交互；就绪与可操作需要分开判断。',featured=True)
    add('search_typed','已输入 ≠ 搜索已完成','verification',['search_empty','search_typed'],'搜索 RoboClaw，并让页面显示匹配的论文结果。',
        {'image_order':['before','after'],'action':'向 ref=7 搜索输入框输入 RoboClaw','ax_before':'ref=7 textbox Search value=""','ax_after':'ref=7 textbox Search value="RoboClaw"; ref=8 button Search; 没有搜索结果'},
        {'step':choice({'passed':'页面已显示与查询匹配的搜索结果，搜索子任务完成。','progress':'关键词已经输入，但尚未显示搜索结果，只取得了局部进展。','failed':'预期的输入操作没有生效。','unknown':'无法从现有证据确认操作效果。'},'比较动作前后的截图与字段值；不要将“已经输入关键词”等同于“已完成搜索”。')},{'step':'progress'},'把动作有效、子任务完成拆开；这是原框架中高价值的验证节点。',featured=True,pair='search_results')
    add('search_results','结果已出现：确认局部完成','verification',['search_typed','search_results'],'搜索 RoboClaw，并让页面显示匹配的论文结果。',
        {'image_order':['before','after'],'action':'点击 Search（搜索）按钮','ax_after':'heading 3 results for RoboClaw; ref=21 link '+TARGET},
        {'step':choice({'passed':'页面已显示与查询匹配的搜索结果，搜索子任务完成。','progress':'关键词已输入，但尚未出现搜索结果。','failed':'存在明确的搜索失败证据。','unknown':'无法从现有证据确认搜索结果。'},'检查当前搜索子任务是否完成，不要推断后续打开论文或下载已经完成。'),'readiness':READINESS},{'step':'passed','readiness':'ready'},'与“只输入”组成正反对照，避免总选失败或等待也能得高分。')
    for wrong in (True,False):
        add('wrong_paper' if wrong else 'correct_paper','关键词相同，论文却不同' if wrong else '精确标题：目标确实匹配','selection',['paper_wrong' if wrong else 'paper_correct'],'打开目标论文：'+TARGET,
            {'target_title':TARGET,'ax':'heading '+('A Survey of Robotic Foundation Models' if wrong else TARGET)},
            {'identity':IDENTITY},{'identity':'no' if wrong else 'yes'},'标题身份比关键词出现更强；用正反配对展示可靠性。',featured=wrong,pair='correct_paper' if wrong else 'wrong_paper')
    for key,title,kind,events,artifact,label in [
        ('pdf_not_downloaded','看到 PDF ≠ 文件已落盘','pdf',[],{'exists':False},'inspect'),
        ('download_pending','传输 40%：继续等待','download_pending',[{'state':'in_progress','received_bytes':3200000,'total_bytes':8000000}],{'exists':False},'wait'),
        ('download_complete','下载完成：事件与文件闭环','download_done',[{'state':'completed','filename':'RoboClaw.pdf'}],{'exists':True,'filename':'RoboClaw.pdf','pdf_parsed':True,'first_page_title':TARGET,'sha256_verified':True},'done'),
        ('download_failed','网络失败：选择恢复分支','download_failed',[{'state':'failed','reason':'network_error'}],{'exists':False},'retry')]:
        add(key,title,'verification',[kind],'将以下目标论文的有效 PDF 保存到磁盘：'+TARGET,{'browser_events':events,'artifact_evidence':artifact},
            {'next':DONE},{'next':label},'截图、浏览器事件与产物证据联合判断；最终完成仍由框架的文件验证器把关。',featured=key=='pdf_not_downloaded',pair='download_complete' if key!='download_complete' else 'pdf_not_downloaded')
    add('tab_routing','四个候选页：选对新标签','selection',['paper_correct','tab_scholar','tab_github','tab_blank'],'进入以下目标论文的摘要页：'+TARGET,
        {'image_order':['tab_A','tab_B','tab_C','tab_D'],'opened_tabs':[{'id':'tab_A','url':'https://papers.example/abs/2603.11558','title':TARGET},{'id':'tab_B','url':'https://scholar.example/search?q=RoboClaw','title':'Scholar search'},{'id':'tab_C','url':'https://code.example/RoboClaw','title':'Repository'},{'id':'tab_D','url':'about:blank','title':'New tab'}]},
        {'tab':choice({'tab_A':'选择第 1 张图对应的论文摘要标签页。','tab_B':'选择第 2 张图对应的学术搜索结果页。','tab_C':'选择第 3 张图对应的代码仓库页。','tab_D':'选择第 4 张图对应的空白新标签页。','unknown':'无法从这些候选中唯一确定目标标签页。'},'只从给定标签页中选择目标论文的摘要页，不得编造候选之外的标签。')},{'tab':'tab_A'},'多图、多候选的有限集合决策；直接返回候选 ID 可避免后续文本解析。',featured=True,pair='tab_ambiguous')
    add('tab_ambiguous','两个空白页：保留 unknown','selection',['tab_unknown','tab_unknown'],'从新打开的标签页中选择以下论文的摘要页：'+TARGET,
        {'image_order':['tab_A','tab_B'],'opened_tabs':[{'id':'tab_A','url':'about:blank','title':'Loading...'},{'id':'tab_B','url':'about:blank','title':'Loading...'}]},
        {'tab':choice({'tab_A':'选择第 1 个仍在加载的空白标签页。','tab_B':'选择第 2 个仍在加载的空白标签页。','unknown':'现有画面和元数据不足以区分哪个标签页才是目标。'},'如果两个标签页的画面和元数据都无法区分，应保留不确定，不要猜测。')},{'tab':'unknown'},'unknown 是明确的业务选项；与低分触发的自动弃权分别展示。',featured=True,pair='tab_routing')
    for id,title,kind,ax,target in [
        ('canvas_gui','有 AX，目标仍在 Canvas','canvas','ref=1 heading Workflow canvas; ref=2 canvas; 没有画布节点的可用引用','画布中的“Export PDF（导出 PDF）”节点'),
        ('pdf_toolbar_gui','PDF 工具栏：切到 GUI','pdf','已提供 PDF 正文文本；浏览器 PDF 工具栏没有可用的控件引用','浏览器 PDF 工具栏中的“Save（保存）”图标'),
        ('native_save_gui','系统另存为：识别桌面表面','native','浏览器 AX 仍对应底层文档；没有提供系统对话框控件的引用','系统“Save As（另存为）”对话框中的“Save（保存）”按钮')]:
        add(id,title,'routing',[kind],'点击下一步目标：'+target,{'ax':ax,'next_target':target},{'route':ROUTE},{'route':'gui'},'路由围绕具体目标，不是“有 AX 就走 DOM”；与原框架的坐标和表面边界一致。',featured=id=='canvas_gui',pair='arxiv_dom_gui')
    add('checkbox_unchanged','点击已派发，勾选没有生效','verification',['checkbox'],'启用自动备份功能。',
        {'executor_status':'dispatched','ax_after':'ref=24 checkbox Enable automatic backups checked=false'},
        {'effect':choice({'yes':'“自动备份”复选框已勾选，预期效果已经发生。','no':'“自动备份”复选框仍未勾选，预期效果没有发生。','unknown':'无法确认复选框当前是否勾选。'},'执行器成功派发点击，不等于界面状态已经改变；以动作后的复选框状态为准。')},{'effect':'no'},'直接展示“执行成功”与“效果成功”的差别。')
    add('focus_wrong','文字输入到了错误字段','verification',['form','focus_wrong'],'在“Project name（项目名称）”输入框中填入 RoboClaw。',
        {'image_order':['before','after'],'action':'输入文字 RoboClaw','ax_before':'Project name value=RoboClaw; Contact email value=team@example.org','ax_after':'Project name value=RoboClaw; Contact email value=RoboClaw focused=true'},
        {'effect':choice({'correct':'指定的项目名称字段按要求发生了变化。','wrong_target':'文字被输入了其他字段，实际操作目标错误。','unknown':'无法确定文字实际输入到了哪个字段。'},'对比前后字段身份与内容，不要只检查页面中是否出现了目标文字。')},{'effect':'wrong_target'},'前后图与字段身份联合验证，发现“文本出现了但目标错了”。')
    add('typed_noul','布尔判断：改动是否已保存','typed',['form_saved'],'判断当前项目设置的全部改动是否已经保存。',
        {'ax':'status All changes saved; dirty=false'},
        {'saved':{'type':'noul','instructions':'结合页面的保存提示和 dirty 标志，判断当前全部改动是否已保存。','criteria':{'true':'全部改动已保存，没有待保存的编辑内容。','false':'仍有尚未保存的改动。'}}},{'saved':'true'},'展示 noul 的原生布尔分数；分数并不等于经过校准的正确率。',pair='typed_noul_false')
    add('typed_noul_false','布尔对照：仍有未保存改动','typed',['form'],'判断当前项目设置的全部改动是否已经保存。',
        {'ax':'status Unsaved changes; dirty=true'},
        {'saved':{'type':'noul','instructions':'结合页面的保存提示和 dirty 标志，判断当前全部改动是否已保存。','criteria':{'true':'全部改动已保存，没有待保存的编辑内容。','false':'仍有尚未保存的改动。'}}},{'saved':'false'},'与已保存场景使用完全相同的问题和判定标准，检查正反一致性。',pair='typed_noul')
    add('typed_score','有序评分：搜索推进到哪一档','typed',['search_results'],'评估“搜索 RoboClaw、打开目标论文并保存其 PDF”这一任务目前推进到了哪一步。',
        {'ax':'Search results for RoboClaw; exact paper link visible','artifact_evidence':{'exists':False}},
        {'progress':{'type':'score','instructions':'仅评估当前可观察的进度。选择更高等级，必须同时满足此前所有阶段的条件。','criteria':['尚未输入搜索关键词。','已输入关键词，但尚未显示搜索结果。','已显示匹配的搜索结果，但尚未打开目标论文。','已打开目标论文，但尚未保存有效的 PDF 文件。','目标论文的有效 PDF 已保存到磁盘并通过校验。']}},{'progress':'2'},'展示 score 概率分布和期望值；评测按最高概率等级，避免拿期望小数强行比类别。')
    return out

CASES=build_cases()
BY_ID={c['id']:c for c in CASES}
MULTI_IDS=['search_typed','search_results','pdf_not_downloaded','download_complete']

@lru_cache(maxsize=64)
def image_data(name):
    return 'data:image/png;base64,'+base64.b64encode((ASSETS/(name+'.png')).read_bytes()).decode()

def evidence(c):
    return {'state':json.dumps(c['state'],ensure_ascii=False,separators=(',',':')),'questions':c['questions'],
            'images':[image_data(k) for k in c['assets']]}

def evidence_hash(c):
    return hashlib.sha256(json.dumps(evidence(c),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def public_case(c):
    return {**c,'images':['/jev/compare/assets/'+k+'.png' for k in c['assets']], 'evidence_hash':evidence_hash(c)}
