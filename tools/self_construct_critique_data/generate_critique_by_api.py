import json
import os
from multiprocessing import Process
import time
from tqdm import tqdm
from typing import Callable
from openai import OpenAI
from pathlib import Path
import argparse
from datasets import load_dataset


def process_chunk(model_name: str,
                  start_idx: int,
                  end_idx: int,
                  input_path: str,
                  output_dir: str,
                  prompt_func: Callable,
                  process_id: int):
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{start_idx + 1}-{end_idx}.json')

    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = {item['idx']: item for item in json.load(f)}

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_data = data[start_idx:end_idx]

    client = OpenAI()

    results = []

    for item in tqdm(chunk_data, desc=f'Process {process_id}'):
        if item['idx'] in existing_results:
            results.append(existing_results[item['idx']])
            continue

        try:
            messages = prompt_func(item)

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=3200,
                top_p=0.95
            )
            item['model_output'] = completion.choices[0].message.content
            item['cost'] = completion.usage.completion_tokens * 10 / 1e6 + completion.usage.prompt_tokens * 2.5 / 1e6
            results.append(item)

            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {item['idx']}: {str(e)}")
            continue
            # item['error'] = str(e)
            # results.append(item)

        time.sleep(0.1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_large_dataset(model_name: str,
                          input_path: str,
                          output_path: str,
                          prompt_func: Callable,
                          num_processes: int = 1):
    temp_output_dir = "temp_output_dir/"
    os.makedirs(temp_output_dir, exist_ok=True)
    if input_path in ["WebInstruct-CFT-50K"]:
        dataset = load_dataset("TIGER-Lab/WebInstruct-CFT", input_path)
        data_list = dataset['train'].to_list()
        data = []
        for each in data_list:
            input_str = each["input"]
            question = input_str.split("\n\nSolution:\n")[0]
            if question.startswith("Question:\n"):
                question = question[len("Question:\n"):]
            answer = input_str.split("\n\nSolution:\n")[1]
            data.append({"question": question, "answer": answer})
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    total_items = len(data)
    print("len(data)", len(data))
    # add idx
    for i, each in enumerate(data):
        if "idx" not in each:
            data[i]["idx"] = i
    with open(input_path, "w") as fo:
        fo.write(json.dumps(data, indent=4))

    chunk_size = total_items // num_processes + 1

    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else total_items

        p = Process(
            target=process_chunk,
            args=(model_name, start_idx, end_idx, input_path, temp_output_dir, prompt_func, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes completed!")
    print("Merging results")
    for file in os.listdir(temp_output_dir):
        if not file.endswith(".json"):
            continue
        output_data = []
        with open(os.path.join(temp_output_dir, file), "r") as fi:
            curr_data = json.load(fi)
            output_data += curr_data
        with open(output_path, "w") as fo:
            fo.write(json.dumps(output_data, indent=4))


def example_prompt_func(item):
    query = f"""
Question: {item["question"]}

Solution: {item["answer"]}
"""
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a science expert. A student is trying to solve a question, please explain briefly whether his solution is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                }
            ]
        },
    ]
    return chat_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process large dataset with multiple processes')
    parser.add_argument('--model_name', type=str, default="gpt-4o-2024-11-20",
                        help='Critique model to use (default: gpt-4o-2024-11-20)')
    parser.add_argument('--input_path', type=str, default="WebInstruct-CFT-50K",
                        help='Path to the input dataset file (default: WebInstruct-CFT-50K)')
    parser.add_argument('--output_path', type=str, default="critique_added_output_data.json",
                        help='Path to store output files (default: ./critique_added_output_data.json)')
    parser.add_argument('--num_processes', type=int, default=20,
                        help='Number of processes to use (default: 20)')

    args = parser.parse_args()

    # os.makedirs(args.output_dir, exist_ok=True)
    process_large_dataset(
        model_name=args.model_name,
        input_path=args.input_path,
        output_path=args.output_path,
        prompt_func=example_prompt_func,
        num_processes=args.num_processes
    )

