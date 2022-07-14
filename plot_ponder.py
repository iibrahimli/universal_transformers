import matplotlib.pyplot as plt
import numpy as np


CB1 = '#dbe9f6'
CB2 = '#bad6eb'
CB3 = '#89bedc'
CB4 = '#539ecd'
CB5 = '#2b7bba'
CB6 = '#0b559f'


labels = ['4', '40', '80', '160', '200', '400']
avg_pond_rev = []
avg_pond_add =  []
avg_pond_copy = []

acc_rev = []
acc_add = []
acc_copy = []

seq_acc_rev = []
seq_acc_add = []
seq_acc_copy = []




x = np.arange(len(labels))*0.2  # the label locations
y = np.arange(0,11)
width = 0.1  # the width of the bars

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(16,9))
rects1 = ax1.bar(x, acc_add, width, color= CB1, label='Addition')
rects2 = ax2.bar(x, acc_copy, width, color= CB3, label='Copying')
rects3 = ax3.bar(x, acc_rev, width, color= CB5, label='Reversing')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Sequence length')
ax1.set_title('Accuracy for "Addition" on different sequence lengths')
ax1.set_xticks(x)
ax1.set_yticks(y)
ax1.set_ylim(0,10)
ax1.set_xticklabels(labels)
#ax1.legend()

ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Sequence length')
ax2.set_title('Accuracy for "Copying" on different sequence lengths')
ax2.set_xticks(x)
ax2.set_yticks(y)
ax2.set_ylim(0,10)
ax2.set_xticklabels(labels)
#ax2.legend()

ax3.set_ylabel('Accuracy')
ax3.set_xlabel('Sequence length')
ax3.set_title('Accuracy for "Reversing" on different sequence lengths')
ax3.set_xticks(x)
ax3.set_yticks(y)
ax3.set_ylim(0,10)
ax3.set_xticklabels(labels)
#ax3.legend()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
#
#
autolabel(rects1, ax1)
autolabel(rects2, ax2)
autolabel(rects3, ax3)

fig.tight_layout()


plt.savefig("acc.pdf")
plt.close()





x = np.arange(len(labels))  # the label locations
y = np.arange(0,11)/10
width = 0.35  # the width of the bars

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(16,9))
ax4=ax1.twinx()
ax5=ax2.twinx()
ax6=ax3.twinx()
rects1 = ax4.bar(x - width/2, seq_acc_add, width, color=CB1, label='seq acc')
rects4 = ax1.bar(x + width/2, acc_add, width, color=CB2, label='char acc')
rects2 = ax5.bar(x - width/2, seq_acc_copy, width, color=CB3, label='seq acc')
rects5= ax2.bar(x + width/2, acc_copy, width, color=CB4, label='char acc')
rects3 = ax6.bar(x - width/2, seq_acc_rev, width, color=CB5, label='seq acc')
rects6 = ax3.bar(x + width/2, acc_rev, width, color=CB6, label='char acc')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Character accuracy')
ax4.set_ylabel('Sequence accuracy')
ax1.set_xlabel('Sequence length')
ax1.set_title('Character and Sequence accuracy for "Addition" for different sequence lenghts')
ax1.set_xticks(x, labels)
ax1.set_yticks(y)
ax1.set_ylim(0,1)
ax4.set_ylim(0,1)
lines1, labels1 = ax1.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax4.legend(lines1 + lines4, labels1 + labels4, loc=0)




ax2.set_ylabel('Character accuracy')
ax5.set_ylabel('Sequence accuracy')
ax2.set_xlabel('Sequence length')
ax2.set_title('Character and Sequence accuracy for "Copying" for different sequence lenghts')
ax2.set_xticks(x, labels)
ax2.set_yticks(y)
ax2.set_ylim(0,1)
ax5.set_ylim(0,1)
lines2, labels2 = ax2.get_legend_handles_labels()
lines5, labels5 = ax5.get_legend_handles_labels()
ax5.legend(lines2 + lines5, labels2 + labels5, loc=0)

ax3.set_ylabel('Character accuracy')
ax6.set_ylabel('Sequence accuracy')
ax3.set_xlabel('Sequence length')
ax3.set_title('Character and Sequence accuracy for "Reversing" for different sequence lenghts')
ax3.set_xticks(x, labels)
ax3.set_yticks(y)
ax3.set_ylim(0,1)
ax6.set_ylim(0,1)
lines3, labels3 = ax3.get_legend_handles_labels()
lines6, labels6 = ax6.get_legend_handles_labels()
ax6.legend(lines3 + lines6, labels3 + labels6, loc=0)

autolabel(rects1, ax1)
autolabel(rects2, ax2)
autolabel(rects3, ax3)
autolabel(rects4, ax4)
autolabel(rects5, ax5)
autolabel(rects6, ax6)

fig.tight_layout()

plt.savefig("acc_acc.pdf")
plt.close()



