import os
import argparse
import json
import re
from tqdm import tqdm
from openai import OpenAI
from multiprocessing import Pool, cpu_count

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='./playground/Instructions_slim/ImageNet/test.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_normaltrain_testslim/ImageNet/OCRVQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/CoIN_normaltrain_testslim/ImageNet/OCRVQA')
    return parser.parse_args()


def eval_single(test_file, result_file):
    annotations = json.load(open(test_file))
    answers = [test['answer'] for test in annotations]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    false_answers = []
    answer_gt_file = []
    for index in tqdm(range(total)):
        text = answers[index]
        label = results[index]
        if (text.upper() in label['text'].upper()) or (label['text'].upper() in text.upper()):
            right += 1
        else:
            label['ground_truth'] = text
            false_answers.append(label)
        answer_gt_file.append({
        "pred": label['text'],
        "ground_truth": text
        })
    ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
    with open(ans_gt_file, "w", encoding="utf-8") as f:
        json.dump(answer_gt_file, f, ensure_ascii=False, indent=4)
        
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
            json.dump(false_answers,f,indent=4)

    return ans_gt_file

def process_batch(api_key, batch):
    """
    对一个批次的数据进行评分，返回该批次所有样本的评分列表。
    """
    from openai import OpenAI  # 确保每个子进程加载必要的模块

    client = OpenAI(api_key=api_key, base_url="https://platform.llmprovider.ai/v1")

    message = (
        "Below are the model's predictions and the ground truth answers for a task. "
        "For each case, provide a semantic similarity score between 0 and 10 in the format 'Score: X', "
        "where X is your score. Always use the format 'Score:' and do not explain anything."
        "\n\nResults:\n" +
        "\n".join([f"{i+1}. Pred: {item['pred']}, Ground Truth: {item['ground_truth']}" for i, item in enumerate(batch)])
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI assistant evaluating a model's prediction quality"}, {"role": "user", "content": message}],
        stream=False
    )

    evaluation_text = response.choices[0].message.content

    # 提取评分
    scores = []
    for line in evaluation_text.splitlines():
        score = float(line.split(":")[1].strip())
        scores.append(score)
    return scores

def deepseek_chat_final(api_key, path, batch_size=10):
    """
    使用多进程评估，返回所有样本的最终平均准确率。
    """
    # 加载 JSON 文件
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 分批处理
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_batches = len(batches)

    print(f"Total data: {len(data)}, Total batches: {num_batches}, Batch size: {batch_size}")

    # 使用多进程池处理所有批次
    total_score = 0
    total_samples = 0
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_batch, [(api_key, batch) for batch in batches]
        )

        # 累计每个批次的评分总和和样本数
        for batch_scores in results:
            total_score += sum(batch_scores)
            total_samples += len(batch_scores)

    # 计算总体评分平均值
    overall_average_score = total_score / total_samples if total_samples > 0 else 0
    return overall_average_score

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        ans_gt_file = eval_single(args.test_file, args.result_file)

        # api_key = "sk-GdmqmU6fFWv5N0HlvYluLFzIbXIPNg3MHzPGeeV247092807Ba2e4487B9D5796cA3Be7dD4"

        # batch_size = 8 
        # overall_accuracy = deepseek_chat_final(api_key, ans_gt_file, batch_size=batch_size)
        # print(f"Overall Accuracy: {overall_accuracy*10:.2f}")
        # if args.output_dir is not None:
        #     output_file = os.path.join(args.output_dir, 'Result_api.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Accuracy: {:.2f}%\n'.format(overall_accuracy*10))
