import os
import json
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import matplotlib.pyplot as plt


def correlation(var1, var2):
    return np.corrcoef(var1, var2)[0, 1]

def correlation2(var1, var2):
    return (np.corrcoef(var1, var2)[0, 1]) ** 2

def spearman(var1, var2):
    from scipy import stats
    return stats.spearmanr(var1, var2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The metric performance.')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--metric', default='IM', type=str)
    parser.add_argument('--pretrained', action='store_true', default=False)

    args = parser.parse_args()
    print(vars(args))

    pretrained = args.pretrained
    metric = args.metric
    dataset = args.dataset

    result_root = r'/home/shuxuan/public_data/Estimate_Performance_RAP/results'

    if metric == "GdScore":
        result_root = os.path.join(result_root, '_GdScore_result')
        GdScore_files = [file for file in os.listdir(result_root) if file.endswith('json')]

        val_scores, val_accs = [], []
        #--------------------------------------------------------------------
        val_type = []
        # -------------------------------------------------------------------
        test_data = defaultdict(list)
        for f in GdScore_files:
            result_file = os.path.join(result_root, f)
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    cur_result = json.load(f)
    
                    val_score, val_acc = -1 * cur_result[0]['val_GdScore'], cur_result[0]['val_acc']
                    val_scores.append(val_score)
                    val_accs.append(val_acc)
                    # ---------------------------------------------------------------------
                    model_name = cur_result[0]['model'].split('_')[0]       # base model, except random seed
                    val_type.append((cur_result[0]['dataset'], model_name))
                    # ---------------------------------------------------------------------
                    cur_test_seen = set()
                    for data in cur_result:
                        if isinstance(data, dict):
                            unique_key = (data.get("test_type"), round(float(data['test_GdScore']), 8))
                            if unique_key in cur_test_seen:
                                continue
                            cur_test_seen.add(unique_key)
                            dataset, category = data["dataset"], data["test_category"]
                            test_data[(dataset, category, model_name)].append((-1 * data['test_GdScore'], data['test_acc']))
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
        # ------------------------------------------------------------------------------
        unique_type = list(set(val_type))
        # color_map = {pr: plt.cm.tab10(i) for i, pr in enumerate(unique_type)}

        lr_relationship= {}
        for pr in unique_type:
            plt.figure(figsize=(8, 6))
            mask = [(d, m) == pr for d, m in val_type]
            x = np.array(val_scores)[mask]
            y = np.array(val_accs)[mask]

            # plt.scatter(x, y, color=color_map[pr], label=pr, alpha=0.6, s=20)
            # plt.xlabel('GdScore(-)')
            # plt.ylabel('validation Acc')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.grid(True, alpha=0.3)
            # plt.tight_layout()
            # plt.savefig(f'/home/shuxuan/code/RAAP/{pr[0]}_{pr[1]}.png')
            # plt.close()

            print()
            X_val_cur = list(x)
            y_val_cur = list(y)
            print(f"{pr[0]}_{pr[1]}")
            print("Correlation:{}".format(correlation2(X_val_cur, y_val_cur)))
            print("Spearman:{}".format(spearman(X_val_cur, y_val_cur).correlation))
            
            if len(X_val_cur) > 1:
                X_val_cur_np = np.array(X_val_cur).reshape(-1, 1)
                y_val_cur_np = np.array(y_val_cur)
                linear_model = LinearRegression().fit(X_val_cur_np, y_val_cur_np)
                lr_relationship[pr] = linear_model
            else:
                raise ValueError(f"Linear relationship can not be built")
        # ------------------------------------------------------------------------------

        unique_result = defaultdict(list)
        for key in test_data:
            dataset, test_category, model = key
            gdScores, test_accs = zip(*test_data[key])
            
            X_test = np.array(gdScores).reshape(-1, 1)
            lr_key = (dataset, model)
            y_pred = lr_relationship[lr_key].predict(X_test)
            y_true = np.array(test_accs)
            
            mae = np.mean(np.abs(y_true - y_pred))
            unique_result[(dataset, test_category)].append(mae)
        for d, c in unique_result:
            mean_mae = np.array(unique_result[(d, c)]).mean()
            print(f"Dataset: {d}, Test Category: {c}, MAE: {mean_mae}")
    else:
        if pretrained:
            sub_dir = "pretrained_ts"
        else:
            sub_dir = "scratch_ts"

        result_path = os.path.join(result_root, dataset, sub_dir, metric)
        type_dir = os.listdir(result_path)

        for type in type_dir:
            cur_dir = os.path.join(result_path, type)
            files = [f for f in os.listdir(cur_dir) if f.endswith('.json')]
            mae_values = []
            for file in files:
                result_file = os.path.join(cur_dir, file)
                cur_result = []
                seen = set()
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        cur_result = json.load(f)
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")

                if not cur_result:
                    continue

                for item in cur_result:
                    if isinstance(item, dict) and 'MAE' in item:
                        unique_key = (item.get("test_type"), round(float(item['MAE']), 8))
                        if unique_key in seen:
                            continue
                        seen.add(unique_key)
                        try:
                            mae_value = float(item['MAE'])
                            mae_values.append(mae_value)
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid MAE value found: {item.get('MAE')}")
                            continue
            # MAE
            true_mae = sum(mae_values) / len(mae_values)
            print(f"{dataset} {type} MAE: {true_mae}")
            # Variance
            std_mae = np.array(mae_values).std()
            print(f"{dataset} {type} MAE STD: {std_mae}")








