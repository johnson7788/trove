"""Online Function Induction and Program Solution Generation."""

import os
import math
import torch
import random
import argparse
from tqdm import tqdm
import transformers
from utils import *
from mako.template import Template
from transformers import AutoTokenizer


def main():
    output = subprocess.run(['which', 'python3'], capture_output=True)
    print(f"要使用的运行LLM输出的工具的python环境是:(确保exec.py中的subprocess.run的配置是python3): \n{output}")
    # load dataset and prompt templates
    dataset = load_dataset(args.task_name, args.max_num_examples)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(dataset)
    # prompt templates, mako的模板语法，就是把模板中的变量进行替换
    create_path = os.path.join("prompt", args.task_name, "online_create.md")
    template_create = Template(filename=create_path)
    import_path = os.path.join("prompt", args.task_name, "online_import.md")
    template_import = Template(filename=import_path)
    skip_path = os.path.join("prompt", args.task_name, "online_skip.md")
    template_skip = Template(filename=skip_path)

    if '/' in args.task_name:
        args.task_name = args.task_name.split('/')[0]
    # library， 最开始，每个toolbox下的每个项目都是空的，例如这里的'toolbox/math.py'是空的
    library_path = os.path.join("toolbox", f"{args.task_name}.py")
    default_library = load_toolbox(library_path)  # 加载了2遍，可以deep COPY啊
    print(f"默认的工具箱中的工具数量: {len(default_library)}")
    library = load_toolbox(library_path)
    print(f"开始加载模型{args.model_name}，可能耗时较长")
    # configure generation pipeline， 加载模型'codellama/CodeLlama-7b-Instruct-hf'
    pipeline = transformers.pipeline(
        "text-generation", model=args.model_name,
        torch_dtype=torch.float16, device_map="auto",
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id  #把padding token赋值成eos token, eos_token_id:2, pad_token_id这里已经是2了，为什么这么做？
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    stable_gen_args = {
        "num_return_sequences": args.num_return_sequences,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    print(f"模型{args.model_name}加载完成")
    fw_log = open(args.output_log_path, 'w')

    def get_example_responses(
        example: dict, index: int, template: Template, library: dict,
    ) -> list[dict]:
        """Get model responses [solution + function(s)] for an example. 每种方式的模板不同"""
        # input
        prompt_args = PROMPT_ARGS_FUNC[args.task_name](example)
        if len(library) > 0 or args.task_name.startswith("math"):
            prompt_args["toolbox"] = format_toolbox(library)
            prompt = template.render(**prompt_args)
            write_prompt(fw_log, prompt, prompt_args["toolbox"], index)
        else:
            prompt = template.render(**prompt_args)
            write_prompt(fw_log, prompt, "", index)

        # output
        max_tokens = len(tokenizer(prompt)["input_ids"]) + args.max_new_tokens
        response_list = pipeline(
            prompt, do_sample=True, max_length=max_tokens, **stable_gen_args
        )
        resp_dict_list = [] #一次回答生成的答案可能有多个
        for r in response_list:
            r = extract_llama_response(r["generated_text"], input_text=prompt)
            resp_dict_list.append(parse_model_response(r))

        #对模型生成的代码进行执行验证结果
        for j, res in enumerate(resp_dict_list):
            # collect code pieces
            code_pieces = []
            for _, func_dict in library.items():
                code_pieces.append(func_dict["function"])
            for func_dict in res["function"]:
                code_pieces.append(func_dict["function"])
            code_pieces.append(unwrap_code(res["solution"]))
            code_pieces = clean_import(code_pieces)  #去掉import的行

            #执行和验证
            is_success, exec_output = execute_code_wrapped(
                code_pieces=code_pieces,
                exec_file=args.exec_file,
                timeout=args.exec_timeout,
            )
            if "answer" in ex:
                answer = ex["answer"]
            elif "answers" in ex:
                answer = ex["answers"]
            else:
                raise ValueError(f"Invalid example w/o answers: {ex.keys()}")
            is_correct, model_answer = EVAL_FUNC[args.task_name](
                is_success=is_success, model_output=exec_output,
                answer=answer, return_answers=True,
            )
            exec_dict = {
                "is_success": is_success,
                "is_correct": is_correct,
                "exec_output": exec_output,
                "model_answers": model_answer,
                "answer": answer,
            }

            # update results, log, and library
            resp_dict_list[j].update(exec_dict)
            write_exec_result(fw_log, exec_dict, index=j) #记录1条日志
            write_solution_and_tools(fw_log, res, library, update_toolbox=False, index=j)

        return resp_dict_list


    def update_library(
        function_list: list[dict], library: dict, match_old: bool = False
    ) -> dict:
        """Update library with function usage or creation."""
        for func_dict in function_list:
            func_name = func_dict["name"]
            if func_name.startswith("toolbox."):
                func_name = func_name[8: ]
            if func_name not in library:
                library[func_name] = func_dict
                library[func_name]["indices"] = [i]
                library[func_name]["frequency"] = 1
            elif match_old and (func_name in library):
                library[func_name]["indices"].append(i)
                library[func_name]["frequency"] += 1
        return library


    def multi_way_generation(
        example: dict, index: int,
        modes: list[str] = ["import", "create", "skip"]
    ) -> dict:
        """Multi-way generation of selected modes."""
        candidate_list = []
        if "import" in modes:  # import模式， 只导入和使用
            print(f"开始尝试第{index}个样本的import模式解决现有问题")
            import_resp_list = get_example_responses(
                example, index, template_import, library
            ) #获取模型的运行结果
            best_import_index = select_best_solution(import_resp_list)  #选中最好的解决方式
            candidate_list.append(import_resp_list[best_import_index]) #加入候选
            print(f"模型import模式一共提出了: {len(import_resp_list)}种解决方案，和答案匹配正确的有: {len([i for i in import_resp_list if i['is_correct']])}个")
        if "create" in modes:
            print(f"开始尝试第{index}个样本的create模式解决现有问题")
            create_resp_list = get_example_responses(
                example, index, template_create, default_library
            ) #获取模型的运行结果
            best_create_index = select_best_solution(create_resp_list)
            candidate_list.append(create_resp_list[best_create_index])
            print(f"模型create模式一共提出了: {len(create_resp_list)}种解决方案，和答案匹配正确的有: {len([i for i in create_resp_list if i['is_correct']])}个")
        if "skip" in modes:
            print(f"开始尝试第{index}个样本的skip模式解决现有问题")
            skip_resp_list = get_example_responses(
                example, index, template_skip, default_library
            )
            best_skip_index = select_best_solution(skip_resp_list)
            candidate_list.append(skip_resp_list[best_skip_index])
            print(f"模型skip模式一共提出了: {len(skip_resp_list)}种解决方案，和答案匹配正确的有: {len([i for i in skip_resp_list if i['is_correct']])}个")

        best_resp_index = select_best_solution(candidate_list)
        best_mode = modes[best_resp_index]  #eg: 'create',表示create模式生成的解决方案效果最好
        best_resp = candidate_list[best_resp_index]

        if best_mode == "import":
            update_library(best_resp["function"], library, match_old=True)
        if (best_mode == "create") and (best_resp["is_success"]):
            update_library(best_resp["function"], library, match_old=False) # 更新工具库

        return {"mode": best_mode, "response": best_resp}


    def trim_library(n: int, library: dict) -> dict:
        """Trimming low-frequency functions from the library，删掉低频率使用的工具"""
        threshold = math.log(n, 20)
        print(
            f"Trimming library of size #{len(library)}",
            f"Usage frequency threshold: {threshold:.2f}",
        )
        for name,d in library.items():
            print(name, " | ", d["frequency"])
            if d["frequency"] < threshold:
                for idx in d["indices"]: trimmed_indices.add(idx)
        library = {name: d for name,d in library.items() if d["frequency"]>=threshold}
        print(f"To size #{len(library)}")
        return library


    # start streaming examples
    result_list = []
    trimmed_indices = set()

    for i, ex in enumerate(tqdm(dataset,desc="进度")):
        # multi-channel (3-way) generation, 加载3种生成工具的方式
        result_dict = multi_way_generation(
            example=ex, index=i,
            modes=["import", "create", "skip"]
        )
        result_list.append(result_dict)

        # periodic forgetting
        if (i + 1) % args.trim_steps == 0:
            library = trim_library(i + 1, library)

    # final forgetting
    library = trim_library(len(dataset), library)

    correct_list = [r["response"]["is_correct"] for r in result_list]
    acc = sum(correct_list) / len(correct_list)
    print(f"整体回复准确率: {acc:.2f}")
    print(f"现有工具箱的大小: #{len(library)}")
    fw_log.write(f"\n## 整体回复准确率: {acc:.2f}\n")
    fw_log.write(f"现有工具箱的大小: #{len(library)}")


    # update solutions of examples missing tools
    trimmed_indices = sorted(list(trimmed_indices))
    print(f"Re-generate solutions for #{len(trimmed_indices)} examples.")
    for i in trimmed_indices:
        result_dict = multi_way_generation(dataset[i], i, ["import", "skip"])
        result_list[i] = result_dict  # update result record

    correct_list = [r["response"]["is_correct"] for r in result_list]
    acc = sum(correct_list) / len(correct_list)
    print(f"经过Trim更新过期工具后，最终回复准确率: {acc:.2f}")

    fw_log.write(f"\n## 经过Trim更新过期工具后，最终回复准确率: {acc:.2f}\n")
    fw_log.write(f"现有工具箱的大小: #{len(library)}")
    for name, d in library.items():
        fw_log.write(f"=== {name} ===\n")
        fw_log.write(d["function"])
        fw_log.write('\n\n\n')
    fw_log.close()
    print(f"保存所有的模型输出和工具输出")
    dump_json_file(result_list, args.output_results_path)
    dump_toolbox(library, args.output_library_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data config
    parser.add_argument("--task_name", type=str, required=True,
                        choices=[
                            "math/algebra", "math/counting", "math/geometry",
                            "math/intermediate", "math/number",
                            "math/prealgebra", "math/precalculus",
                            "tabmwp", "wtq", "hitab", "gqa"
                        ],
                        help="Task name.")
    parser.add_argument("--shuffle_seed", type=int, default=None)
    
    # experiment config
    parser.add_argument("--run_index", type=int, default=None)

    # example config
    parser.add_argument("--max_num_examples", type=int, default=None,
                        help="Maximum number of examples to generate.")
    parser.add_argument("--trim_steps", type=int, default=500,
                        help="Trim library by threshold every N examples.删除低频率使用的工具，每隔多少个step检查1次工具库")

    # execution config
    parser.add_argument("--exec_file", type=str, default="tmp_exec_online.py",
                        help="LLM生成的代码保存到临时执行文件后执行")
    parser.add_argument("--exec_timeout", type=int, default=100,
                        help="Timeout for execution in seconds.")

    # generation config
    parser.add_argument("--model_name", type=str, 
                        default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=256,help="最多生成的新的token数量")

    args = parser.parse_args()
    args.suffix = "trove"
    args = auto_decide_path(args, fields=["library", "log"])

    main()
