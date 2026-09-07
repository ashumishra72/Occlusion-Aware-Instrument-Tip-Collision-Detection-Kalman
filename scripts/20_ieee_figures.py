"""IEEE-style method and result figures: serif text, compact labels, PNG/SVG."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/charts/updated'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'svg.fonttype':'none','axes.linewidth':.65,'savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def pipeline():
    fig,ax=plt.subplots(figsize=(7.16,3.45))
    fig.subplots_adjust(0,0,1,1); ax.set_xlim(0,7.16); ax.set_ylim(0,3.45); ax.axis('off')
    def box(x,y,w,h,title,desc,fill='#f4f4f4'):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor='black',lw=.7))
        ax.text(x+w/2,y+h-.12,title,ha='center',va='top',weight='bold',fontsize=8.3)
        ax.text(x+w/2,y+.10,desc,ha='center',va='bottom',fontsize=7.7,linespacing=1.25)
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.7,
            linestyle='--' if dashed else '-',color='black',shrinkA=1,shrinkB=1))
    box(.04,2.53,1.13,.70,'Video','One recording\n3,917 sampled frames')
    box(1.45,2.53,1.47,.70,'YOLO11n-seg','TIPL / TIPR / marker\nBoxes, masks, confidence')
    box(3.20,2.53,1.54,.70,'Kalman tracking','One filter per tip\nPrediction, age, covariance')
    box(5.02,2.53,2.10,.70,'Features','Original 43 / Kalman (53)\nKalman + 2 s history (85)')
    for a,b in [((1.17,2.88),(1.45,2.88)),((2.92,2.88),(3.20,2.88)),((4.74,2.88),(5.02,2.88))]: arrow(a,b)
    # Explicit original-feature branch: original features do not pass through Kalman.
    ax.plot([2.18,2.18,6.07],[3.23,3.35,3.35],color='black',lw=.65,ls='--')
    arrow((6.07,3.35),(6.07,3.23),True)
    ax.text(4.12,3.39,'Original 43-feature branch (bypass)',ha='center',va='bottom',fontsize=7)
    ax.text(3.58,2.32,'Cached detector measurements are reused; no detector retraining in these comparisons.',
        ha='center',fontsize=7.5,style='italic')
    box(.04,1.24,1.60,.75,'Annotations','64 collision rows\nFrame labels: +/-0.5 s')
    box(1.96,1.24,2.40,.75,'Grouped model evaluation','24 groups; 5 outer folds; 4 s guard\nRF / XGBoost; inner validation')
    box(4.70,1.24,2.42,.75,'Score to collision events','Causal 3-frame mean\nValidation-selected threshold + gap')
    arrow((1.64,1.61),(1.96,1.61)); arrow((4.36,1.61),(4.70,1.61))
    arrow((6.07,2.53),(6.07,2.08)); arrow((6.07,2.08),(3.15,2.08)); arrow((3.15,2.08),(3.15,1.99))
    box(.04,.08,3.12,.72,'Two extraction objectives','Frame-F1 threshold + fixed 0.4 s gap\nEvent-F1 threshold + gap search',fill='white')
    box(3.53,.08,3.59,.72,'Held-out evaluation','Frame metrics + missing-tip subset\nOne-to-one event matching: detected / missed / false',fill='white')
    arrow((3.16,.44),(3.53,.44)); arrow((5.91,1.24),(5.91,.80))
    ax.text(3.58,1.01,'Inner validation chooses settings. Outer test labels are used only for evaluation.',
        ha='center',fontsize=7.5,style='italic')
    save(fig,'fig1_ieee_end_to_end_pipeline')

def results():
    old=json.loads((ROOT/'results/kalman_updated/metrics.json').read_text())
    tuned=json.loads((ROOT/'results/event_f1_tuned/metrics.json').read_text())
    names=list(old); labels=['O-RF','O-XGB','K-RF','K-XGB','KT-RF','KT-XGB']
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.88))
    x=np.arange(6); width=.35
    colors=['#ffffff','#526779']; hatches=['///','']
    settings=[('frame','(a) Frame-level F1'),('event','(b) Event-level F1')]
    for ax,(metric,title) in zip(axes,settings):
        a=np.array([old[n][metric]['f1'] for n in names]); b=np.array([tuned[n][metric]['f1'] for n in names])
        for values,xx,color,hatch in [(a,x-width/2,colors[0],hatches[0]),(b,x+width/2,colors[1],hatches[1])]:
            bars=ax.bar(xx,values,width,facecolor=color,edgecolor='black',linewidth=.55,hatch=hatch)
            for bar,value in zip(bars,values):
                ax.text(bar.get_x()+bar.get_width()/2,value+.008,f'{value:.3f}',ha='center',va='bottom',rotation=90,fontsize=6.5)
        ax.set_ylim(0,.52); ax.set_yticks([0,.1,.2,.3,.4,.5]); ax.set_xticks(x,labels,rotation=35,ha='right')
        ax.set_ylabel('F1 score'); ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.22,lw=.5); ax.set_axisbelow(True)
    handles=[Rectangle((0,0),1,1,facecolor=colors[0],edgecolor='black',hatch='///',lw=.55),
             Rectangle((0,0),1,1,facecolor=colors[1],edgecolor='black',lw=.55)]
    fig.legend(handles,['Frame-F1 tuning; fixed gap','Event-F1 tuning; searched gap'],loc='lower center',
        bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.07,right=.995,bottom=.28,top=.88,wspace=.23)
    save(fig,'fig2_ieee_frame_event_results')

if __name__=='__main__': pipeline(); results(); print(OUT)
