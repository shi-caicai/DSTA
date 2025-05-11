import pandas as pd
import re
from sklearn.metrics import f1_score

def cal_result(eval_output):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output], ignore_index=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    df = df[['pred', 'label', 'id']]

    # Create a new column to store adjusted predictions
    df['adjusted_pred'] = df['pred']

    # compute accuracy
    correct = 0
    y_true = []
    y_pred = []

    for i, row in df.iterrows():
        pred = row['pred']
        label = row['label']

        matches = re.findall(r"real|Real|Fake|fake", pred.strip())
        y_true.append(label)

        if len(matches) <= 0:
            if label == 'fake':
                df.at[i, 'adjusted_pred'] = 'real'
                y_pred.append('real')
            elif label == 'real':
                df.at[i, 'adjusted_pred'] = 'fake'
                y_pred.append('fake')
        else:
            adjusted_value = matches[0].lower()
            df.at[i, 'adjusted_pred'] = adjusted_value
            y_pred.append(adjusted_value)

        if len(matches) > 0 and adjusted_value == label:
            correct += 1

    # Update the 'pred' column with adjusted predictions
    df['pred'] = df['adjusted_pred']
    df.drop(columns='adjusted_pred', inplace=True)

    # 定义替换映射关系
    mapping = {'real': 0, 'fake': 1}
    # 利用列表推导式替换元素
    y_true_replaced = [mapping[label] for label in y_true]
    y_pred_replaced = [mapping[label] for label in y_pred]

    macro_f1 = f1_score(y_true_replaced, y_pred_replaced, average='macro')
    f1_fake = f1_score(y_true_replaced, y_pred_replaced)

    # 定义替换映射关系
    mapping = {'real': 1, 'fake': 0}
    # 利用列表推导式替换元素
    y_true_replaced = [mapping[label] for label in y_true]
    y_pred_replaced = [mapping[label] for label in y_pred]

    f1_real = f1_score(y_true_replaced, y_pred_replaced)

    return correct / len(df) , f1_real, f1_fake, macro_f1
