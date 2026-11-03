from pathlib import Path
import json,csv,subprocess,hashlib
from html import escape
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'
read=lambda name:json.loads((A/name).read_text())
D=read('E2_GO_NO_GO.json');F=read('frozen/frozen_summary.json');B=read('feedback/feedback_summary.json');G=read('groups/geometry_summary.json')
M=read('groups/mapping_manifest.json');FEAT=read('groups/feature_definition.json');TEST=read('tests_result.json');PROD=read('production_checks.json')
PROPOSAL=read('minimal_frozen_supplement_proposal.json');SPANS=read('frozen/observed_group_span_summary.json');ELIG=read('feedback/window_eligibility_summary.json')
VERS=read('E2_source_versions.json')
OUT=ROOT/'output/pdf/E2_WBC_序列聚类与共享统计_离线报告_20261104.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CJK','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
W,H=landscape(A4);margin=34;CW=W-2*margin;NAVY=colors.HexColor('#18394a');TEAL=colors.HexColor('#107c8a');GREY=colors.HexColor('#546877');LIGHT=colors.HexColor('#eef4f7')
c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('E2 - WBC 序列聚类与共享 SGS 统计：离线报告');c.setAuthor('Codex');page=0
def p(text,x,y,width=CW,size=10,leading=None,color=NAVY):
    s=ParagraphStyle('p',fontName='CJK',fontSize=size,leading=leading or size*1.5,textColor=color,wordWrap='CJK',splitLongWords=True)
    q=Paragraph(escape(str(text)).replace('\n','<br/>'),s);_,h=q.wrap(width,H);q.drawOn(c,x,y-h)
    assert y-h>=34,('paragraph overflow',page,str(text)[:60],y-h)
    return y-h
def title(text,subtitle=''):
    global page
    page+=1;c.setFillColor(TEAL);c.rect(0,H-9,W,9,fill=1,stroke=0)
    p(text,margin,H-28,CW,20,25)
    if subtitle:p(subtitle,margin,H-60,CW,9,13,GREY)
def footer():
    c.setStrokeColor(colors.HexColor('#d1dee5'));c.line(margin,30,W-margin,30);c.setFont('CJK',8);c.setFillColor(GREY)
    c.drawString(margin,17,'E2 | WBC | 2026-11-04 | 原六组15k保持只读 | 本轮正式 PPO 更新数 = 0');c.drawRightString(W-margin,17,f'{page} / 12');c.showPage()
def section(text,y):return p(text,margin,y,CW,13,18,TEAL)-7
def table(rows,widths,y,size=9,x=margin):
    sty=ParagraphStyle('cell',fontName='CJK',fontSize=size,leading=size*1.4,textColor=NAVY,wordWrap='CJK',splitLongWords=True)
    t=Table([[Paragraph(escape(str(v)),sty) for v in r] for r in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9ef')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#bfd0d9'))]))
    _,h=t.wrap(sum(widths),H);assert y-h>=35,('table overflow',page,y,h);t.drawOn(c,x,y-h);return y-h
def figure(path,top,width=CW,height=370):
    im=ImageReader(str(path));iw,ih=im.getSize();s=min(width/iw,height/ih);w=iw*s;h=ih*s;c.drawImage(im,(W-w)/2,top-h,width=w,height=h,mask='auto');return top-h
def get_window(run,mapping):return next(x for x in B['window_statistics'] if x['run']==run and x['mapping']==mapping and x['checkpoint']==15000)
def get_elig(run,mapping):return next(x for x in ELIG['rows'] if x['run']==run and x['mapping']==mapping)
def pred(est,first,key='Brier'):return next(x[key] for x in B['prediction_rows'] if x['estimator']==est and x['first_update']==first)
def frozen(cp,mode,mapping,metric):return next(x for x in F['rows'] if x['policy']=='WBC-SGS-E1' and x['checkpoint']==cp and x['action_mode']==mode and x['mapping']==mapping and x['metric']==metric)
def comp(cp,mode,metric):return next(x for x in F['primary_comparisons'] if x['checkpoint']==cp and x['action_mode']==mode and x['metric']==metric and x['random_mapping']=='random_20261104')
cmd=next(x for x in G['rows'] if x['mapping']=='command');randoms=[x for x in G['rows'] if x['mapping']!='command'];seq=get_window('uniform','sequence');group=get_window('uniform','command')

#1 Research decision and evidence classes.
title('E2 - WBC 序列聚类与共享 SGS 统计','结论：NO-GO，暂缓正式 PPO；先补全簇完整冻结评测。所有分组、统计实现、测试和离线分析均已完成。')
y=p('共享反馈确实更及时，Command Cluster 的前向预测也有小幅收益；但现有固定策略数据未证实它能稳定形成控制表现更一致的簇。用户“GO 后自动启动两组各10k”的授权条件尚未满足，因此本轮没有启动正式训练。',margin,H-87,CW,11,17)-13
rows=[['预先固定的条件','结果','证据'],['工程正确性','通过',f"{TEST['tests']} 项单元测试；4套真实32768->4096映射的Torch checkpoint、归属与RNG恢复检查通过。"],['命令相似性','通过',f"平均簇内距离 {cmd['mean_feature_distance']:.4f}，三个Random为 {min(x['mean_feature_distance'] for x in randoms):.4f}-{max(x['mean_feature_distance'] for x in randoms):.4f}。"],['固定策略控制一致性','未通过','位置RMSE和联合达标时间的簇内差异，未在两个checkpoint、两种动作模式下稳定优于全部3个Random；区间跨过无差异。'],['历史状态估计诊断','通过','在同一Uniform反馈流上，Command的前向Brier低于3个Random和逐序列估计；这不等于新PPO收益。']]
y=table(rows,[138,70,CW-208],y,9)-15
y=section('事实、推断、未验证假设',y)
y=p(f"已验证：保留32,768条训练序列和独立1,024条评测序列；4套映射均为4,096x8；反馈窗口仍是100个真实结束回合。Uniform15k时，平均记录年龄从 {seq['mean_feedback_age_updates']:.0f} 降至 {group['mean_feedback_age_updates']:.0f} 次更新。",margin,y,CW,10,15)-8
y=p('合理推断：逐序列历史未填满100时，较早策略留下的结果长期保留，可能拖慢能力估计更新。共享窗口有缓解这一问题的证据。',margin,y,CW,10,15)-8
p('未验证：新分组能否提高从零训练的联合成功率、能否在全部8个成员中保持难度同质、能否跨训练seed复现。',margin,y,CW,10,15)
footer()

#2 Pool schema and actual native bounds.
title('A - 原序列池字段与实际命令范围','原训练池 wbc_train.npz；EE float32[32768,4,7]；velocity float32[32768,4,5]；固定生成 seed721102。')
y=H-89
rows=[['数组字段','原始含义与实际映射','运行语义'],['EE[...,0:3]','x=[0.45,0.70]m；y=[-0.35,0.35]m；z=[0.10,0.80]m','x/y在link0系，z是世界高度；不能把三者范数当机械臂伸展半径。'],['EE[...,3:6]','roll扰动 +/-3.14/3；pitch扰动 +/-3.14/4；yaw扰动 +/-3.14/6 rad','先按机器人状态计算朝向，叠加扰动；三个角统一clamp到[-3.14/4,+3.14/3]，再转四元数。'],['EE[...,6]','重采样间隔 = 8+2u 秒','完整20秒回合中仅前两个间隔会影响后续有效EE目标。'],['velocity[...,0:3]','vx=[-1,1]m/s；vy=[-0.6,0.6]m/s；wz=[-1,1]rad/s','本体速度命令；heading_command=false。'],['velocity[...,3]','u <= 0.1 为站立指令','常规compute后速度置零；显式初始reset有一帧尚未mask的例外。'],['velocity[...,4]','映射到固定[10,10]秒','随机值本身无效，不能作为聚类距离输入。'],['其他条件','EE curriculum=false；velocity curriculum=false；保留原域随机化','序列只固定命令随机输入；初态、摩擦、质量、关节reset等未写死在sequence ID内。']]
y=table(rows,[111,308,CW-419],y,9)-13
y=p('同一sequence ID不等于预先写死的世界坐标轨迹。特别是WBC目标姿态仍依赖当前link0状态；过去日志没有保存每次重采样的完整最终四元数与物理状态，不能声称恢复了全训练过程的实际姿态分布。',margin,y,CW,9.5,14)-9
p('本轮重新检查命令读取、reset、command compute、回合反馈及原SGS FIFO。原始数组共有48个uniform数，但部分槽和计时随机列不会影响20秒回合；特征按执行语义筛选。',margin,y,CW,9.5,14)
footer()

#3 Exact timing and strict labels.
title('A - 真正执行的槽：初始 reset 与自动 reset 有区别','16环境只读追踪覆盖两个完整回合；精确float32计时模型与32,000条控制步样本逐项匹配。')
y=H-88
rows=[['条件','EE目标','Velocity目标'],['显式初始reset，完整20秒','slot0起始；slot1约8-10s；slot2约16-20s。若到达终止边界则不再执行旧序列的新命令。','slot0为前500个控制步；slot1为后500步；slot2不执行。'],['原生自动reset后的回合','reset所在帧先执行一次compute；有效切换可比初始回合提前1个控制步。','slot0从索引0开始；slot1从索引499开始；slot2从索引999开始，最后20ms执行一次。'],['提前失败或启动时随机episode length','实际执行前缀变短，不能保证3个EE槽都执行。','实际前缀同样可变短；不能利用旧失败结果截断聚类输入，否则会泄漏表现。'],['结构上无效的内容','EE slot3全部字段；EE slot2的timer。','velocity slot3全部字段；所有velocity timer随机值。slot2的一帧内容必须保留。']]
y=table(rows,[140,312,CW-452],y,9)-13
y=p('全32768条序列、完整自动reset回合的命令覆盖：EE slot0/1/2分别占44.93% / 45.06% / 10.01%的控制步；velocity slot0/1/2占49.9% / 50.0% / 0.1%。三个有效速度槽的站立比例为9.86% / 10.10% / 9.70%。EE两次切换时刻的中位数为8.98s / 18.00s，四分位区间分别为8.48-9.48s / 17.42-18.58s。',margin,y,CW,9.2,13.5)-11
y=section('成功标签保持原定义，回填发生在 reset 清空指标之前',y)
y=p('5cm / 80%：速度误差 <=0.20m/s，偏航速度误差 <=0.30rad/s，末端位置误差 <=0.05m，姿态误差 <=pi/12。两类命令年龄均满2秒才计分，至少100个有效步，四项同时达标时间 >=80%，各项RMSE合格，且没有原生失败。超时本身不代表成功。',margin,y,CW,9.5,14)-10
y=p('原SequenceAdapter在reward计算后、原生_reset_idx前读取终止批次；按pool、sequence、env和generation验证归属，再提交一次反馈。自动reset最后一帧的新速度命令仍遵循原生计分窗口，本轮不修正或替换原指标。',margin,y,CW,9.5,14)-10
p('v1只追踪初始回合，曾遗漏自动reset的一帧velocity slot2。源码计时模型发现后，追加第二回合只读追踪并重建v2特征、映射与分析；v1已归档排除。没有按成功表现搜索特征权重。',margin,y,CW,9,13,GREY)
footer()

#4 Features and equal-capacity algorithm.
title('B - 命令表征与等容量聚类','只读取训练命令与共享reset参考状态；没有成功率、学习进展或独立评测数据输入。')
y=H-88
rows=[['特征块','维数','处理'],['位置 / 速度时间轨迹','120 / 120','40个固定0.5s时间箱；按原生物理范围归一化，保留顺序和站立mask。'],['站立轨迹','40','保留特殊模式；随机uniform值只保留其站立布尔结果。'],['SO(3)目标与扰动','720','共享reset参考状态下的原生目标旋转 + 姿态扰动旋转；使用旋转矩阵弦距离，不使用四元数分量欧氏距离。'],['精确切换 / 初始例外','2 / 3','EE切换时刻按20s缩放；初始站立命令的一帧raw速度单独保留。'],['精确驻留事件','75','保留末段短事件，含自动reset最后一帧速度slot2；按真实驻留时长加权。'],['合计',FEAT['dimensions'],'字段和权重在最终映射生成前固定，未按控制表现或GO结果调优。']]
y=table(rows,[172,72,CW-244],y,9)-13
y=p('SO(3)关系：||R1-R2||_F^2 = 8*sin(theta/2)^2，对四元数正负号不敏感，并随旋转角单调变化。几何对照另报告精确测地角theta。参考状态来自加载策略前的reset，link0高度约0.42m；扰动旋转保留后续状态依赖的输入信息。',margin,y,CW,9.5,14)-11
y=p('等容量方法：在归一化命令特征上，递归取近似主轴并按中位数等分；每个叶子恰好8条，得到4096组。每节点8次power iteration，seed20261104，投影并列按sequence ID稳定排序。不是无容量约束KMeans后再任意塞入剩余序列。',margin,y,CW,9.5,14)-11
p('限制：固定参考姿态不能代表机器人运动后的全部目标四元数。当前距离是命令相似性的可复现近似，不是可达性证明，也不保证沿控制难度相关方向聚得最近。',margin,y,CW,9,13,GREY)
footer()

#5 Geometry.
title('B - 命令确实更接近，但几何结果不能替代控制验证','每套映射覆盖全部32,768条训练序列，4096组x8条；每组28个无序对，总114,688对。')
y=figure(A/'groups/command_geometry.png',H-83,height=230)
rows=[['映射','平均归一化距离','参考SO(3)角距离','位置归一化绝对差']]
for r in G['rows']:rows.append([r['mapping'],f"{r['mean_feature_distance']:.4f}",f"{r['mean_reference_SO3_geodesic_deg']:.2f} deg",f"{r['mean_normalized_position_abs_difference']:.4f}"])
y=table(rows,[174,195,195,CW-564],y-9,9)-12
reduction=100*(1-cmd['mean_feature_distance']/randoms[0]['mean_feature_distance'])
y=p(f"相对主Random，Command的整体簇内距离降低 {reduction:.1f}%。位置差异和姿态差异也降低，但幅度不同；这说明聚类满足命令相似性设计目标，尚不能推导出成功率估计或PPO必然改善。",margin,y,CW,9.5,14)-8
p('三套Random均由固定seed随机排列后每8条一组；不选择离线表现最好或最差的Random做主对照。主对照seed20261104在分析前固定。',margin,y,CW,9,13,GREY)
footer()

#6 Fixed policy consistency and coverage.
title('C - 冻结策略一致性：区间仍宽，不能确认控制同质性','复用E1 CP4000/8000/10000和原SGS CP8000/10000，每种均值/原生探索模式各1024条训练序列x4次reset。')
y=H-88
coverage=F['coverage']['E1_10000_mean'];rows=[['映射','有>=2已评成员的簇','已观测簇内对','完整8成员簇']]
for name,cov in coverage.items():rows.append([name,cov['multi_member_groups'],cov['within_group_pairs'],cov['fully_observed_groups']])
y=table(rows,[194,214,184,CW-592],y,8.8)-12
rows=[['CP / 动作','位置差异变化 mm [95%区间]','联合时间差异变化 pp [95%区间]']]
for cp in (8000,10000):
    for mode in ('mean','stochastic'):
        a=comp(cp,mode,'position');b=comp(cp,mode,'joint_fraction')
        rows.append([f'{cp} / {mode}',f"{a['command_minus_random']:+.2f} [{a['ci95_low']:.2f}, {a['ci95_high']:.2f}]",f"{b['command_minus_random']:+.2f} [{b['ci95_low']:.2f}, {b['ci95_high']:.2f}]"])
y=table(rows,[174,300,CW-474],y,8.8)-11
y=p('表中是“Command簇内绝对差 - 主Random簇内绝对差”，负值表示更同质。已知分层抽样使用成对纳入概率的归一化逆概率权重；共享序列节点Poisson bootstrap共2000次。原SGS面板仅作其历史选择条件下的补充分析。全部3个Random比较另附CSV。',margin,y,CW,9,13)-9
s=SPANS['mean'];p(f"CP10000均值动作下，{s['partial_groups']}个可观察的部分Command簇中，{s['observed_success_span_at_least50pp']}个的经验成功率跨度 >=50pp，{s['observed_position_span_at_least10mm']}个的位置RMSE跨度 >=10mm。每序列仅4次reset，不能把这些跨度直接称为真实难度差异。",margin,y,CW,9,13)
footer()

#7 Fixed policy curves and progress.
title('C - 固定策略下的组内差异与学习进展','位置/连续时间指标仅用计分成熟且无原生失败的试次；严格成功率仍包含全部试次。')
y=figure(A/'frozen/frozen_consistency.png',H-80,width=720,height=300)
y=section('学习进展的一致性：簇内 |成员进展差|，均值动作',y-7)
rows=[['冻结区间','Command位置差 mm','Random位置差 mm','Command联合时间差 pp','Random联合时间差 pp']]
for interval in ('4000-8000','8000-10000'):
    def val(mapping,metric):return next(x['mean_within_group_abs_difference'] for x in F['rows'] if x['policy']=='WBC-SGS-E1-progress' and x['checkpoint']==interval and x['action_mode']=='mean' and x['mapping']==mapping and x['metric']==metric)
    rows.append([interval,f"{val('command','position'):.2f}",f"{val('random_20261104','position'):.2f}",f"{val('command','joint_fraction'):.2f}",f"{val('random_20261104','joint_fraction'):.2f}"])
y=table(rows,[112,159,159,172,CW-602],y,8.5)-8
p('这是同一固定面板在两个checkpoint间的表现变化差异。它不证明更多采样造成了进展，也不覆盖整个训练池；全量95%区间和三套Random结果见配套表。',margin,y,CW,8.5,12,GREY)
footer()

#8 Density and age.
title('C - 反馈密度提高、历史年龄降低：已有直接日志证据','完整审查Uniform、原SGS、E1的0-15k真实反馈；历史统计来自变化中的策略，不当作冻结难度。')
y=figure(A/'feedback/feedback_density_age_prediction.png',H-82,height=230)
rows=[['Uniform15k估计器','窗口平均回合数','平均反馈年龄 / 更新','95%年龄 / 更新','满100窗口']]
for name in ('sequence','random_20261104','command'):
    r=get_window('uniform',name);rows.append([name,f"{r['mean_window_samples']:.2f}",f"{r['mean_feedback_age_updates']:.0f}",f"{r['p95_feedback_age_updates']:.0f}",f"{r['full_windows']}/{r['groups']}"])
y=table(rows,[166,150,190,161,CW-667],y-10,8.7)-12
e0=get_elig('uniform','sequence');e1=get_elig('uniform','command')
y=p(f"逐序列窗口中，{100*e0['feedback_end_before4000_fraction']:.2f}%的记录来自4k及以前；计分不足比例 {100*e0['insufficient_window_fraction']:.2f}%，原生失败比例 {100*e0['native_failure_fraction']:.2f}%（可重叠）。Command近期窗口中，计分不足为 {100*e1['insufficient_window_fraction']:.3f}%，原生失败为 {100*e1['native_failure_fraction']:.3f}%。",margin,y,CW,9.5,14)-10
p('分组每单位收到8个成员的反馈，但FIFO仍只保留100次真实回合；并非保留800个样本。上表年龄按窗口内记录加权。Random与Command的新鲜度几乎相同，因此新鲜度收益主要来自共享粒度。没有删除任何失败或计分不足标签。',margin,y,CW,9,13,GREY)
footer()

#9 Prediction and mixing.
title('C - 估计更及时，但仍会损失成员难度差异','同一Uniform事件流，先预测下一个真实严格标签，再加入反馈；没有重采样或更新策略。Brier越低越好。')
y=H-87
rows=[['估计器','10,001-12,500 Brier','12,501-15,000 Brier','去除自身反馈后的后段Brier']]
for name in ('sequence','random_20261104','random_20261105','random_20261106','command'):
    rows.append([name,f"{pred(name,10001):.6f}",f"{pred(name,12501):.6f}",'无同组其他成员' if name=='sequence' else f"{pred(name,12501,'leave_member_out_Brier'):.6f}"])
y=table(rows,[170,199,199,CW-568],y,8.8)-12
y=p('Command在两个后段块中都优于三套Random；100-update块bootstrap的95%区间均排除零。相对逐序列Brier降低约2.6%。但去除本条序列自身反馈后，只靠组内其他成员预测更差，说明成员差异仍重要，不能仅凭更低的混合Brier宣称簇已同质。',margin,y,CW,9.3,14)-12
rows=[['遗留日志','同窗口成员权重差距 / pp','单成员最大占比中位数','遗漏成员的簇数']]
for run in ('uniform','sgs','sgs-e1'):
    m=next(x for x in B['same_window_member_mixture'] if x['run']==run and x['mapping']=='command');w=get_window(run,'command')
    rows.append([run,f"{100*m['mean_abs_same_window_member_weighting_gap_observed_members']:.2f}",f"{100*w['median_largest_member_share']:.0f}%",m['missing_member_groups']])
y=table(rows,[161,249,224,CW-634],y,8.8)-10
p('差距比较同一last100窗口的“按回合数加权成功率”和“已观察成员等权均值”，仍含有限样本噪声，不是真实偏差的直接测量。原SGS/E1采样不均，不能把旧日志直接当成新分组会产生的反馈。另列的近期组均值与终生成员均值之差还混入策略年龄，已明确分开。',margin,y,CW,8.8,13,GREY)
footer()

#10 Sampler and tests.
title('D - 同一套共享 SGS 代码，唯一实验差异是映射','code/grouped_sampling.py + code/grouped_adapter.py；WBC原奖励、观测、18D动作、终止、随机化及PPO均保持原样。')
y=H-87
for text in ['p_k = S_k / N_k；S/N均来自第k组最近100个真实结束回合；空历史p_k=0.5，N=0。',
             'w(p)=(p+eps)^(c*target) * (1-p+eps)^(c*(1-target))；f(p)=exp(log(max(w(p),eps)+eps)/temperature)。',
             'Q(k)=n_k*f(p_k)/sum_j[n_j*f(p_j)]；选组后在真实成员中均匀抽取，P(i|k)=1/n_k。',
             '所有组权重相同时，每条sequence的概率严格相等，为1/32768；不等组大小的通用形式也已测试。']:
    y=p(text,margin,y,CW,10,15)-7
y=section('反馈和 checkpoint',y-3)
y=p('沿用原生终止前指标读取；按pool、env、sequence、generation拒绝错配或重复。共享FIFO保留成员来源、控制步和事件身份，按原生终止顺序更新；不把整个回合反复记账。checkpoint保存映射/池哈希、组窗口、计数、反馈年龄来源、事件代数与RNG。评测走独立evaluation适配器，不更新分组历史。',margin,y,CW,9.5,14)-12
rows=[['验证范围','结果'],['单元测试',f"{TEST['tests']}项通过：核公式、归一化、空先验、FIFO100、大小修正、时序/无效字段、SO(3)符号、重复/外来反馈、评测隔离。"],['保存恢复','4套真实32768->4096映射，4096-env归属模拟，Torch保存/加载、后续4096抽样逐项相同；JSON/RNG往返和损坏RNG原子拒绝。'],['基线与范围','原6个15k checkpoint哈希未变；原源码未变；工程合成历史仅在tests_only，正式PPO更新数为0。'],['尚未验证','没有运行新采样器的4096环境PPO短测或正式训练；CPU采样计时不代表完整在线吞吐。']]
y=table(rows,[147,CW-147],y,9)-10
p('参数保持 history100、target0.5、concentration1、temperature2、epsilon1e-4。两个未来训练组均须从零初始化模型、Adam和历史，不能加载本轮工程测试的checkpoint。',margin,y,CW,9,13,GREY)
footer()

#11 Decision and supplement.
title('E - NO-GO：先补簇完整冻结验证，再决定正式训练','工程、命令几何和前向估计条件通过；固定策略一致性条件未通过。当前不启动两个10k。')
y=H-88
y=p('现有训练池面板虽然覆盖多个checkpoint和两种动作模式，但没有任何完整8成员簇，单条只有4次reset。逆概率加权后有效对数进一步下降，主要连续差异区间跨过零。当前证据支持更及时的估计，尚不足以支持完整PPO预算。',margin,y,CW,10,15)-13
rows=[['最小冻结补充方案（已生成选集，未执行）','固定内容'],['选择',f"seed20261107；64个随机anchor，对Command和3套Random各补齐64个完整8成员簇；union共{PROPOSAL['unique_training_sequences']}条训练序列。"],['策略与模式','原WBC-Uniform CP10000、CP15000；均值动作与原生训练探索动作，权重冻结。'],['重置与预算',f"每条至少2次相同配对reset；每个应用{PROPOSAL['episodes_per_app']}试次，4个应用，合计{PROPOSAL['total_episode_trials']:,}试次。两次重复是最小筛查，必要时预先约定追加同面板2次以改善二项精度。"],['严格边界','原5cm/80%及原噪声/DR/终止规则；相同序列与reset面板供全部映射复用；不训练、不回填历史、不读独立评测池。'],['主要问题','组内位置/时间/严格成功差异是否更小，是否跨checkpoint与动作模式稳定；用完整成员检查聚合掩盖的难度差异。']]
y=table(rows,[175,CW-175],y,9)-13
y=p('这份补充方案没有执行。用户已授权“若GO则自动启动Random Group SGS与Command Cluster SGS各10k”；当前GO条件未满足，所以没有把该授权当成无条件训练许可。',margin,y,CW,9.5,14)-10
p('若后续证据达到GO，正式对照应保持4096环境、rollout24、seed42、相同初始网络和原生PPO；Random主映射固定20261104，Command固定当前映射，二者从零训练并使用相同独立评测集。',margin,y,CW,9,13,GREY)
footer()

#12 Reproducibility and files.
title('交付索引与可复现版本','原始大数据留YangLab；交付包含完整映射、全部离线指标、图表、源码、测试与未执行的补充选集。')
y=H-87
rows=[['映射','seed','映射 SHA256（int64向量）']]
for name,mm in M['mappings'].items():rows.append([name,mm['seed'],mm['mapping_sha256']])
y=table(rows,[165,83,CW-248],y,8.2)-11
rows=[['文件 / 目录','内容'],['groups/sequence_to_group_all_mappings.csv + mapping_*.npy','全部32768条ID的4套映射；所有组8条。feature_definition.json与mapping_manifest.json固定输入、归一化、种子和哈希。'],['frozen/','簇内控制差异、全部Command对3套Random的95%区间、部分簇跨度、学习进展与固定策略来源哈希。'],['feedback/','窗口密度/年龄、计分不足和原生失败比例、前向Brier、同窗口成员混合、逐序列后段代理。'],['code/ + tests/ + Git bundle','共享采样器、Isaac适配器、等容量分组和离线分析源码；可恢复独立分支e2/grouped-sgs-offline。'],['E2_GO_NO_GO.json / minimal_frozen_supplement_proposal.json','机器可读判定、预注册条件、固定补充评测序列与预算；proposal executed=false。']]
y=table(rows,[251,CW-251],y,8.5)-11
y=p('原生提交 4e12109d1eccb1d9da2e4a35f189fabea3f13677；Isaac Lab b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8。NumPy1.26.0、Torch2.7.0+cu128、IsaacLab0.54.4、IsaacSim5.1.0.0、RSL-RL5.0.1。核心代码提交 bc28815584adfe05c680908557a6f810ac4b351b；完整代码文件哈希见E2_source_versions.json。',margin,y,CW,8.5,12)-8
p('源码线索：audit_sources/native_pose_command_wbc.py:_resample_command；isaaclab_command_manager.py:CommandTerm.reset/compute；isaaclab_manager_based_rl_env.py:step/_reset_idx；code/adapter.py:reward/reset。远端根目录 /data/go2_piper_sgs_e2_20261104；v1修订、逐步追踪、大特征数组、工程checkpoint与原始日志均保留远端。',margin,y,CW,8.5,12,GREY)
footer()

assert page==12;c.save()
print(OUT);print('pages',page,'bytes',OUT.stat().st_size,'sha256',hashlib.sha256(OUT.read_bytes()).hexdigest())
