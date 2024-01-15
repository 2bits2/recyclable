import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Actual': [1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'Predicted': [1, 0, 3, 0, 0, 1, 1, 0, 1, 1]}

df = pd.DataFrame(data)
confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)

heatmap = sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')

plt.title('Confusion Matrix')
#plt.show()

plt.savefig('confusion_matrix.svg', format='svg', bbox_inches='tight')



