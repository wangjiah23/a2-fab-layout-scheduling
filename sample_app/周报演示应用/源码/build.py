"""Build the portable application from the course materials. No solver is run."""
import base64
import csv
import io
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET


SRC = Path(__file__).resolve().parent
APP = SRC.parent
ROOT = APP.parent
EXAMPLES = APP / '导入示例'
TABLES = ROOT / '第五周/smat_layout_viz/smat_layout_viz/tables'
E1 = ROOT / 'week6/output/e1_reference_lp_full'
SMT = ROOT / 'week6/data/SMT2020_LVHM'
FACTORY = 'SMAT2022-LVHM_E'
FULL = 'full'
SETTINGS = {'horizon_min': 10080, 'effective_time_fraction': .85}


def records(p, delimiter=','):
    with p.open(encoding='utf-8-sig' if delimiter == ',' else 'latin1', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def pack(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def b64(raw):
    return base64.b64encode(raw).decode('ascii')


def factory():
    slots, fixed = [], []
    for row in records(TABLES / 'equipment_position_summary.csv'):
        item = dict(equipment_name=row['NAME'], tool_group=row['TOOL_GROUP'], bay=row['BAY'],
                    area=row['area'], x=float(row['POSITION_X']), y=float(row['POSITION_Y']),
                    width=float(row['WIDTH']), height=float(row['HEIGHT']))
        (slots if row['kind'] == 'PRODUCTION' else fixed).append(item)
    rails = [[row['BAY_NAME'], *[round(float(row[k]), 3) for k in ('x0','y0','x1','y1')]]
             for row in records(TABLES / 'rail_geometry.csv')]
    return dict(slots=slots, fixed=fixed, rails=rails)


def scenario_data(summary):
    parts = records(SMT/'part.txt','\t')
    order_rows = records(SMT/'order.txt','\t')
    sizes = {r['PART']:float(r['PIECES']) for r in order_rows if r['PIECES']}
    routes = {}
    for p in parts:
        steps = []
        for r in records(SMT/p['ROUTEFILE'],'\t'):
            t = float(r['PTIME'])
            if r['PTPER'] == 'per_piece':
                t *= sizes[p['PART']]
            elif r['PTPER'] == 'per_batch':
                t *= sizes[p['PART']]/((float(r['BATCHMN'])+float(r['BATCHMX']))/2)
            delay = r['STNFAM'].lower().startswith('delay')
            steps.append([int(r['STEP']), 'delay' if delay else 'process', r['STNFAM'], round(t,9), r['DESC']])
        routes[p['ROUTE']] = steps
    demands = [dict(part=p['PART'],route=p['ROUTE'],demand_lots=summary['product_demands_lots'][p['PART']]) for p in parts]
    return routes, dict(factory_id=FACTORY,scope=FULL,**SETTINGS,demands=demands,new_routes=[])


def workbook(scenario,name):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    b=Workbook(); s=b.active; s.title='产品需求';s.append(['产品编号','路线编号','需求量（lot）'])
    for r in scenario['demands']: s.append([r['part'],r['route'],r['demand_lots']])
    s2=b.create_sheet('新增路线');s2.append(['路线编号','工序序号','工序类型','设备族编号','单批用时（分钟）'])
    for sheet in b:
        sheet.freeze_panes='A2'
        for c in sheet[1]:
            c.font=Font(color='FFFFFF',bold=True)
            c.fill=PatternFill('solid',fgColor='17594F')
            sheet.column_dimensions[c.column_letter].width=24
    buf=io.BytesIO();b.save(buf)
    data=buf.getvalue();(EXAMPLES/name).write_bytes(data)
    return b64(data)


def rl_data():
    from PIL import Image
    with ZipFile(ROOT/'期中汇报/A2_期中汇报.pptx') as ppt:
        tree=ET.fromstring(ppt.read('ppt/charts/chart1.xml'))
        ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        ser=tree.find('.//c:ser',ns)
        def series(key):
            return [float(el.text) for el in ser.findall('c:'+key+'//c:pt/c:v',ns)]
        curve=[[int(x),y*100] for x,y in zip(series('xVal'),series('yVal'))]
        g=Image.open(io.BytesIO(ppt.read('ppt/media/image32.gif')))
        frames=[]
        for frame in sorted({*range(0,g.n_frames,6),g.n_frames-1}):
            g.seek(frame); im=g.convert('RGB');im.thumbnail((1600,600),Image.Resampling.LANCZOS)
            buf=io.BytesIO();im.save(buf,'WEBP',quality=83,method=3)
            frames.append({'frame':frame,'img':b64(buf.getvalue())})
    return dict(curve=curve,frames=frames,epoch=95,frameCount=g.n_frames)


def render_html(data):
    html=(SRC/'shell.html').read_text(encoding='utf-8')
    vals={'STYLE':(SRC/'style.css').read_text(encoding='utf-8'),
          'ZIP_LIB':(SRC/'jszip.min.js').read_text(encoding='utf-8'),
          'DATA':data,
          'MODEL':(SRC/'io.js').read_text(encoding='utf-8'),
          'MAP':(SRC/'map.js').read_text(encoding='utf-8'),
          'UI':'\n'.join((SRC/f).read_text(encoding='utf-8') for f in ['views.js','hosted.js','app.js'])}
    for k,v in vals.items():html=html.replace('<!-- '+k+' -->',v)
    out=APP/'晶圆厂布局工作台.html'
    newline='\r\n' if out.is_file() and b'\r\n' in out.read_bytes()[:100] else '\n'
    out.write_bytes(html.replace('\n',newline).encode('utf-8'))
    return out


def main():
    EXAMPLES.mkdir(exist_ok=True)
    (EXAMPLES/'05_订单.txt').write_bytes((SMT/'order.txt').read_bytes())
    with ZipFile(ROOT/'week7_heuristic.zip') as z:
        prefix='week7_heuristic/code/output/e3_global/'
        raw_layout=z.read(prefix+'layout.csv');metrics=json.loads(z.read(prefix+'e3_summary.json'))
        layout=list(csv.DictReader(io.StringIO(raw_layout.decode('utf-8-sig'))))
    for r in layout:
        for k in ('x','y','width','height'):r[k]=float(r[k])
        for k in ('original_slot','new_slot'):r[k]=int(r[k])
    (EXAMPLES/'03_启发式全厂布局.csv').write_bytes(raw_layout)
    ref=json.loads((E1/'e1_summary.json').read_text(encoding='utf-8-sig'))
    routes,scene=scenario_data(ref)
    changed=json.loads(pack(scene));by={d['part']:d for d in changed['demands']}
    by['part_1']['demand_lots']-=5;by['part_5']['demand_lots']+=5
    templates={'baseline':workbook(scene,'01_基准生产场景.xlsx'),
               'changed':workbook(changed,'02_产品配比变化.xlsx')}
    plant=factory(); group_alias={r['route_stnfam']:r['target_smt_stn'] for r in records(ROOT/'week6/output/e0_audit/route_coverage.csv') if r['target_smt_stn']}
    local_html=(ROOT/'week6/MILP交互展示.html').read_text(encoding='utf8')
    local=json.loads(re.search('type="application/json">(.*?)</script>',local_html,re.S)[1])
    before={r['equipment_name']:r for r in plant['slots']}
    slot_name={r['original_slot']:r['equipment_name'] for r in layout}
    assert len(set(r['new_slot'] for r in layout))==len(before)==metrics['equipment_count']==len(layout)
    for r in layout:
        target=before[slot_name[r['new_slot']]]
        assert all(abs(r[k]-target[k])<.001 for k in ['x','y','width','height']) and r['bay']==target['bay']
    bundle={'schema':'fab-result-v1','scenario':scene,'method':'heuristic_lp','metrics':metrics,'layout':layout}
    (EXAMPLES/'04_启发式离线结果包.json').write_text(pack(bundle),encoding='utf-8')
    archive=dict(plant=plant,routes=routes,scene=scene,changed=changed,templates=templates,
                 aliases=group_alias,e3=dict(metrics=metrics,layout=layout),rl=rl_data(),
                 local=dict(summary=local['summary'],slots=local['slots'],html=b64(local_html.encode())),
                 rawCsv=b64(raw_layout))
    out=render_html(pack(archive).replace('<','\\u003c'))
    package=ROOT/'周报演示应用_离线版.zip'
    source_files=['hosted.js','app.js','build.py','io.js','JSZip-LICENSE.md','jszip.min.js','map.js','shell.html','style.css','views.js']
    examples=['01_基准生产场景.xlsx','02_产品配比变化.xlsx','03_启发式全厂布局.csv','04_启发式离线结果包.json','05_订单.txt']
    with ZipFile(package,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for path in [out,APP/'周报演示与导入说明.md',*[EXAMPLES/f for f in examples],*[SRC/f for f in source_files]]:
            z.write(path,'周报演示应用/'+path.relative_to(APP).as_posix())
    print(out.name,out.stat().st_size,'bytes',len(archive['rl']['frames']),'replay frames')
    print(package.name,package.stat().st_size,'bytes')


if __name__=='__main__':
    if sys.argv[1:] == ['--ui-only']:
        existing=(APP/'晶圆厂布局工作台.html').read_text(encoding='utf-8')
        data=re.search(r'<script id="archive" type="application/json">(.*?)</script>',existing,re.S)
        if not data:raise ValueError('Embedded course archive missing')
        render_html(data[1])
    else:
        main()
