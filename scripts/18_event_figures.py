"""Export editable SVG and high-resolution PNG figures for the event report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
M=json.loads((ROOT/'results/event_f1_tuned/metrics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
NAVY='#18374b'; TEAL='#087f8c'; GRAY='#687b89'; PALE='#eef4f7'; GOLD='#c18932'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=240,facecolor='white',bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),facecolor='white',bbox_inches='tight')
    plt.close(fig)

def roadmap():
    fig,ax=plt.subplots(figsize=(12,8)); ax.set_xlim(0,12); ax.set_ylim(0,8); ax.axis('off')
    ax.text(.15,7.73,'Five ways to improve collision detection',fontsize=23,weight='bold',color=NAVY)
    ax.text(.15,7.26,'One experiment completed now. Four directions still need development or more data.',color=GRAY,fontsize=12)
    rows=[
      ('1','Count whole collisions','Tune the score cutoff and gap between positive samples.','TESTED NOW',TEAL),
      ('2','Learn the movement sequence','Try LSTM, GRU or TCN after adding independent labeled videos.','AFTER MORE DATA',GOLD),
      ('3','Track the two tips together','Model relative motion and shared uncertainty; test an IMM extension.','PROPOSED',GOLD),
      ('4','Keep the correct tip identity','Use appearance or low-confidence detections to reconnect tracks.','PROPOSED',GOLD),
      ('5','Add more labeled videos','Include different trainees, motions, lighting and hidden-tip examples.','DATA NEEDED',GOLD)]
    for i,(num,title,desc,status,col) in enumerate(rows):
        y=5.94-i*1.1
        ax.add_patch(FancyBboxPatch((.15,y),11.65,.93,boxstyle='round,pad=0.02,rounding_size=0.10',
            linewidth=0,facecolor=PALE))
        ax.text(.48,y+.47,num,ha='center',va='center',fontsize=19,weight='bold',color=col)
        ax.text(.98,y+.61,title,fontsize=14,weight='bold',color=NAVY)
        ax.text(.98,y+.22,desc,fontsize=11,color=NAVY)
        ax.text(11.53,y+.62,status,fontsize=9,weight='bold',ha='right',color=col)
    ax.text(.15,.57,'Current test:  same classifier scores  +  validation-selected threshold and gap  ->  collision events',
        fontsize=12,color=TEAL,weight='bold')
    ax.text(.15,.12,'A proposal is not a measured improvement. Compare every extension on held-out video groups.',fontsize=11,color=GRAY)
    save(fig,'figure_improvement_roadmap')

def comparison():
    names=list(M)
    labels=[n.replace('Original43','Original 43').replace('KalmanTemporal','Kalman + history').replace('_',' / ') for n in names]
    old=np.array([M[n]['baseline_event']['f1'] for n in names])
    new=np.array([M[n]['event']['f1'] for n in names])
    fig,ax=plt.subplots(figsize=(11.8,6.8))
    y=np.arange(len(names)); h=.32
    ax.barh(y-h/2,old,height=h,color='#9aacb8',label='Frame-F1 threshold; fixed 0.4 s gap')
    ax.barh(y+h/2,new,height=h,color=TEAL,label='Event-F1 threshold and gap')
    for values,ys in [(old,y-h/2),(new,y+h/2)]:
        for value,yy in zip(values,ys): ax.text(value+.008,yy,f'{value:.3f}',va='center',fontsize=11,color=NAVY)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,.5)
    ax.set_xlabel('Held-out event F1 (higher is better)',labelpad=12,color=NAVY)
    ax.set_title('Event-focused tuning: small gains across all six models',loc='left',fontsize=19,weight='bold',pad=42,color=NAVY)
    ax.text(0,1.035,'Only the threshold and merge gap change. Classifier scores and outer folds stay fixed.',
        transform=ax.transAxes,fontsize=10.5,color=GRAY)
    ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=12)
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.30),frameon=False,ncol=1,fontsize=10)
    fig.subplots_adjust(left=.24,bottom=.22,top=.82,right=.95)
    save(fig,'figure_event_f1_results')

if __name__=='__main__':
    roadmap(); comparison(); print('Saved PNG/SVG figures in',OUT)
