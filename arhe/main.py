import os
import subprocess as sp
import argparse
import re
import time
import sys
import json
import requests
import csv
import ast
from transformers import AutoTokenizer
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import logging

from tqdm import tqdm
import openai

from evaluator import Evaluator

sys.path.append('/data/wyk/bigcodebench/agents')
from evaluate import untrusted_check
from models import QWenLLM
from utils import setup_log, setup_separate_log, extract_code_block, extract_method_source, extract_imports

DEBUG = False
EPHEMERAL_FILE = '__test__.py'

class CommentRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.body = [e for e in node.body if not (isinstance(e, ast.Expr) and isinstance(e.value, ast.Constant))]
        self.generic_visit(node)
        return node

def query_model(prompt, end_tokens=['`'], max_tokens=1000):
    # response = openai.Completion.create(
    #     model='gpt-3.5-turbo-instruct',
    #     prompt=prompt,
    #     temperature=0.0,
    #     max_tokens=max_tokens,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     stop=end_tokens
    # )
    # return_text = response['choices'][0]['text']
    resps, logrpobs = llm.generate(prompt, n=1, temperature=0.0, top_p=1, stop=end_tokens, max_tokens=max_tokens)
    return_text = resps[0]
    return return_text

def get_error_msg_from(test_file):
    working_dir = os.path.dirname(test_file)
    p = sp.run(['python', os.path.basename(test_file)], 
               capture_output=True, cwd=working_dir)
    error_msg = p.stderr.decode('utf-8').strip()
    assert len(error_msg) != 0
    return error_msg

class PDBWrapper():
    def __init__(self, start_cmd):
        self._start_cmd = start_cmd
        self._pdb = sp.Popen(start_cmd.split(), stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
                             cwd=os.getcwd())
        self.client_preamble = self._read_stdout_to_prompt() + '\n(Pdb)'
        self._client_terminated = False
    
    def _read_stdout_to_prompt(self):
        stdout_read = ''
        next_char = 'a' # dummy value
        while ((re.search('\(Pdb\)', stdout_read) is None) and next_char):
            next_char = self._pdb.stdout.read(1).decode()
            stdout_read += next_char
        if not next_char:
            self._client_terminated = True
        stdout_read = stdout_read.strip()
        return '\n'.join(stdout_read.split('\n')[:-1])
    
    def _send_command(self, cmd):
        self._pdb.stdin.write(cmd.encode())
        self._pdb.stdin.write(b'\n')
        self._pdb.stdin.flush()
        out = self._read_stdout_to_prompt()
        return out
    
    def execute_command(self, cmd, with_unroll=False):
        if not with_unroll or 'p ' != cmd.split(';;')[-1].strip()[:2]:
            stdout_read = self._send_command(cmd)
            return stdout_read
        else:
            cmd_blocks = cmd.split(';;')
            cmd_blocks = cmd_blocks[:2] + ['globals().update(locals())'] + cmd_blocks[2:]
            cmd = ';;'.join(cmd_blocks)
            first_output = self._send_command(cmd)
            output_list = [first_output.split('\n')[-1]]
            if 'Uncaught exception.' in first_output:
                return '[The breakpoint line was not covered by the test.]'
            cmd_without_breakpoint = ';;'.join(cmd.split(';;')[1:])
            next_output = self._send_command('c')
            while 'Uncaught exception.' not in next_output:
                next_output = self._send_command(cmd.split(';;')[-1])
                output_list.append(next_output.split('\n')[-1])
                next_output = self._send_command('c')
            
            # error output handling
            if all('***' in e for e in output_list):
                return output_list[0]
            else:
                return [e for e in output_list if '***' not in e]
                
    
    def terminate(self):
        self._pdb.terminate()
    
    @property
    def terminated(self):
        return self._client_terminated

class PromptBuilder():
    def __init__(self, function_file, template_file = 'initial_template.txt', verbose=False, output_file="./task_id/agent_trace.csv"):
        with open(template_file) as f:
            prompt_template = f.read().strip()
        self._error_msg = get_error_msg_from(function_file)
        self._prompt = prompt_template.format(
            function_file = function_file,
            function_code = self._file_numberer(function_file),
            error_msg = self._error_msg
        )
        self._function_file = function_file
        if verbose:
            print(self._prompt, end='')
        self._verbose = verbose
        self._interaction_done = False
        self._cr = CommentRemover()

        # step-wise debug state 
        self.cur_step = 0
        with open(self._function_file) as f:
            file_content = f.read()
            self.cur_code = file_content.split("import unittest")[0]
            lines = file_content.splitlines()
            test_line = next((line for line in lines if 'testcases.test_' in line), None)
            self.cur_test = test_line.split('testcases.')[1].split('(')[0] if test_line else None
        self.cur_nl_plan = ""
        self.cur_debugger_cmd = ""
        self.cur_exec_observation = self._error_msg
        self.cur_conclusion = ""
        self.output_file = output_file

        # llm info
        self.enc = token_enc #= AutoTokenizer.from_pretrained("/data/share/qwen/Qwen2.5-Coder-7B-Instruct/")
        self.input_token = 0
        self.output_token = 0
        self.model_call = 0
    def safe_query_model(self, prompt, end_tokens=['`'], max_tokens=None):
        save_err = None
        for _ in range(5):
            try:
                self.model_call += 1
                self.input_token += len(self.enc.encode(prompt))
                result = query_model(prompt, end_tokens, max_tokens)
                self.output_token += len(self.enc.encode(result))
                return result
            except Exception as e:
                print('ERR:', e)
                save_err = e
                time.sleep(8)
        return f'Error persisted: {str(save_err)}'
    def _simulated_print(self, text):
        for c in text:
            sys.stdout.write(c)
            sys.stdout.flush()
            time.sleep(0.03)

    def _add_to_prompt(self, text):
        self._prompt += text
        if self._verbose:
            self._simulated_print(text)

    def _file_numberer(self, file):
        with open(file) as f:
            lines = f.readlines()
        numbered_lines = []
        for i, line in enumerate(lines):
            numbered_lines.append(f'{i+1} {line}')
        return ''.join(numbered_lines).strip()
    
    def start_interaction(self, append_start = True):
        prehandling = self.safe_query_model(self._prompt, end_tokens=['Attempt'], max_tokens=None)
        self._add_to_prompt(prehandling)
        if append_start:
            self._add_to_prompt('Attempt')
    
    def _start_pdb(self):
        cmd = f'python -m pdb {self._function_file}'

        self._pdbw = PDBWrapper(cmd)
        process_result = self._pdbw.client_preamble
    
    def _terminate_pdb(self):
        self._pdbw.terminate()

    def _replace_in_file(self, replace_line, org_expr, new_expr):
        replace_line = int(replace_line)
        with open(self._function_file) as f:
            org_lines = f.readlines()
        new_line = org_lines[replace_line-1].replace(org_expr, new_expr)
        if new_line == org_lines[replace_line-1]:
            raise ValueError(f'expr {org_expr} not found in line {replace_line}')
        org_lines[replace_line-1] = new_line
        with open(self._function_file, 'w') as f:
            f.write(''.join(org_lines))
        
    def _exec_pdb_command(self, debugger_cmd, with_unroll=True):
        if 'AND RUN' in debugger_cmd:
            if 'REPLACE' not in debugger_cmd:
                return 'Unknown command; please use REPLACE.'
            replace_cmd = debugger_cmd.removesuffix(') AND RUN').removeprefix('REPLACE(')
            try:
                replace_line, org_expr, new_expr = list(csv.reader([replace_cmd], skipinitialspace=True))[0]
            except Exception:
                print(f'FAILURE ON {replace_cmd} ;;')
                return f'Could not parse {replace_cmd}; please specify three arguments.'
            try:
                self._replace_in_file(replace_line, org_expr, new_expr)
                with open(self._function_file) as f:
                    file_content = f.read()
                    self.cur_code = file_content.split("import unittest")[0]
                    lines = file_content.splitlines()
                    test_line = next((line for line in lines if 'testcases.test_' in line), None)
                    self.cur_test = test_line.split('testcases.')[1].split('(')[0] if test_line else None
                error_msg = get_error_msg_from(self._function_file)
                self._replace_in_file(replace_line, new_expr, org_expr) # revert the change
                return error_msg.strip().split('\n')[-1]
            except ValueError as e:
                return str(e)
            except AssertionError:
                self._replace_in_file(replace_line, new_expr, org_expr)
                return '[No exception triggered]'
            except Exception as e:
                print('Unfamiliar exception', e)
                self._replace_in_file(replace_line, new_expr, org_expr)
                return str(e)
        
        output_obj = self._pdbw.execute_command(debugger_cmd, with_unroll=with_unroll)
        if not with_unroll or 'p ' != debugger_cmd.split(';;')[-1].strip()[:2]:
            return output_obj.split('\n')[-1]
        else:
            if isinstance(output_obj, str):
                return output_obj
            else:
                assert type(output_obj) == list
                output_list = output_obj

            if 'AssertionError' in self._error_msg:
                true_debugger_output = output_list[:len(output_list)//2]
            else:
                true_debugger_output = output_list

            if len(true_debugger_output) == 1:
                return str(true_debugger_output[0])
            else:
                return 'At each loop execution, the expression was: ' + '['+', '.join(true_debugger_output)+']'
            
        
    def single_step(self):
        self.cur_nl_plan = self.safe_query_model(self._prompt, end_tokens=['Experiment:'], max_tokens=None)
        self._add_to_prompt(self.cur_nl_plan + 'Experiment: `')
        debugger_cmd = self.safe_query_model(self._prompt, end_tokens=['Observation:'])
        debugger_cmd = extract_code_block(debugger_cmd).replace('`','').strip()
        debugger_cmd = debugger_cmd.replace(' n ;', ' c ;')
        self.cur_debugger_cmd = debugger_cmd.replace(' ; ', ' ;; ')
        self._add_to_prompt(self.cur_debugger_cmd + '`\nObservation: `')
        
        self._start_pdb()
        self.cur_exec_observation = self._exec_pdb_command(self.cur_debugger_cmd)
        self._terminate_pdb()
        
        self._add_to_prompt(self.cur_exec_observation + '`\nConclusion:')
        self.cur_conclusion = self.safe_query_model(self._prompt, end_tokens=['Attempt', '```\n', '```python'], max_tokens=None)
        
    def _content_without_comments(self, fname):
        with open(fname) as f:
            root = ast.parse(f.read())
            last_function_node = [e for e in root.body if isinstance(e, ast.FunctionDef)][-1]
            return ast.unparse(self._cr.visit(last_function_node))
        
    def final_step(self):
        self._trace = self._prompt
        self._prompt = self._redact_failures()
        self._add_to_prompt('The repaired code (full method, without comments) is:\n\n```python\ndef')
        patch = self.safe_query_model(self._prompt, end_tokens=['```\n'], max_tokens=None)
        # augment the patch with imports
        patch = patch.split('```python')[-1].replace('```','').strip()
        imports = extract_imports(self.cur_code)
        for imp in imports:
            if imp not in patch:
                patch = imp + '\n' + patch

        self._add_to_prompt('```python\n' + patch + '```\n')

        # format the result
        self.cur_nl_plan = self._prompt
        self.cur_debugger_cmd = 'final_step'
        self.cur_exec_observation = patch
        self.cur_conclusion = ""
        self.save_step_state()

    def save_step_state(self):
        pd.DataFrame([{
            'step': self.cur_step,
            'file': self._function_file,
            'code': self.cur_code,
            'test': self.cur_test,
            'nl_plan': self.cur_nl_plan,
            'debugger_cmd': self.cur_debugger_cmd,
            'exec_observation': self.cur_exec_observation,
            'conclusion': self.cur_conclusion
        }]).to_csv(self.output_file, index=False, mode='a', header=not os.path.isfile(self.output_file))
    def take_steps(self, n_steps=10):
        self.save_step_state()
        for i in range(n_steps-1): # 预留一个step用来final step
            self.cur_step += 1
            # print("step",self.cur_step)
            self.single_step()
            self.save_step_state()

            if '<DEBUGGING DONE>' in self.cur_conclusion:
                self._add_to_prompt(self.cur_conclusion.split('<DEBUGGING DONE>')[0] + '<DEBUGGING DONE>\n\n')
                self.cur_step += 1
                self.final_step()
                self.save_step_state()
                self._interaction_done = True
            else:
                self._add_to_prompt(self.cur_conclusion)
                if not i == n_steps-2: # 如果是倒数第二步，就不用再加Attempt了
                    self._add_to_prompt('Attempt')
            
            if self._interaction_done:
                break
        
        if not self._interaction_done:
            self.cur_step += 1
            self.final_step()
            self.save_step_state()
            self._interaction_done = True
    
    def _redact_failures(self, split_token = '## Analysis'):
        """Redacts the failed attempts from the prompt. Specifically, removes the failed attempts and the corresponding observations."""
        assert self._prompt.count(split_token) == 1
        attempt_str = self._prompt.split(split_token)[-1]
        attempt_lines = attempt_str.split('\n')
        conclusion_strings = [e for e in attempt_lines if 'Conclusion:' in e]
        failed_attempts = [i+1 for i, e in enumerate(conclusion_strings) if 'is supported' not in e]
        failed_attempt_string_indexes = [f'Attempt {i}.' for i in failed_attempts]
        failed_attempt_starts = [i for i, e in enumerate(attempt_lines) if e in failed_attempt_string_indexes]
        failed_attempt_ends = [i+1 for i, e in enumerate(attempt_lines) if 'Conclusion: The hypothesis is rejected' in e]
        failed_attempt_ends = [0] + failed_attempt_ends
        failed_attempt_starts = failed_attempt_starts + [-1]
        
        new_str = self._prompt.split(split_token)[0] + split_token
        for prev_end, new_start in zip(failed_attempt_ends, failed_attempt_starts):
            new_str += '\n'.join(attempt_lines[prev_end:new_start]) + '\n'
        return new_str
    
    def get_solution(self):
        assert self._interaction_done
        generated_sol = self._prompt.split('```python')[-1].split('```')[0]
        generated_sol = '\n'.join([
            re.sub('^\d+ ', '', e)
            for e in generated_sol.split('\n') 
            if not (e.isnumeric() or 'assert' in e)])
        return generated_sol
        

class ASDEvaluator(Evaluator):
    def __init__(self, mutant_file, template_file):
        with open(mutant_file) as f:
            self._mutant_info = json.load(f)[:5]
        self._template_file = template_file
        super(ASDEvaluator, self).__init__()
    
    def get_solutions(self, N=1, steps=3, ephemeral_file=EPHEMERAL_FILE):
        if N != 1:
            raise NotImplementedError
        for idx, mutant_instance in tqdm(enumerate(self._mutant_info), total=len(self._mutant_info)):
            custom_test = mutant_instance['failed_tests'][0]['failing_assertion'].strip()
            try:
                exec_code = self._get_exec_code(
                    solution = {
                        'task_id': mutant_instance['task_id'],
                        'samples': [mutant_instance['mutant']],
                    }, 
                    custom_test = custom_test,
                    wrap_with_f = False
                )
            except ValueError as e:
                print('Warning:', type(e), e)
                mutant_instance['samples'] = [mutant_instance['mutant']]
                continue
            
            with open(ephemeral_file, 'w') as f:
                f.write(exec_code)
            
            try:
                prompt_builder = PromptBuilder(ephemeral_file, self._template_file, verbose=DEBUG)
                prompt_builder.take_steps(n_steps=steps)
                proposed_sol = prompt_builder.get_solution()
            except Exception as e:
                print('Warning@PromptBuilder:', type(e), e)
                mutant_instance['samples'] = ['']
                continue
            if DEBUG:
                print(proposed_sol)
                print('a-ok')
                exit(0)
            # mutant_instance['result'] = result
            mutant_instance['samples'] = [proposed_sol]
            mutant_instance['trace'] = prompt_builder._trace
            mutant_instance['prompt_at_repair'] = prompt_builder._prompt

            # evalaute one solution at a time, and save to 
            evaluation_object = {
                'task_id': mutant_instance['task_id'],
                'samples': mutant_instance['samples']
            }
            failing_tests = self.evaluate_sol(evaluation_object)
            mutant_instance['passed'] = (len(failing_tests) == 0)
            # TODO: code_status: pass, sys_error, fail, error
            mutant_instance['fail_tests'] = failing_tests
            mutant_instance['ARHE_id'] = idx
            # save to file
            with open(args.output_file, 'a') as f:
                json.dump(mutant_instance, f)
                f.write('\n')

    # TODO: def evaluate_sol(self, solution):
    
    def evaluate_all_solutions(self):
        for idx, mut_info in enumerate(self._mutant_info):
            evaluation_object = {
                'task_id': mut_info['task_id'],
                'samples': mut_info['samples']
            }
            failing_tests = self.evaluate_sol(evaluation_object)
            mut_info['passed'] = (len(failing_tests) == 0)
            mut_info['fail_tests'] = failing_tests
            mut_info['ARHE_id'] = idx
    
    def save_to(self, file_name):
        with open(file_name, 'w') as f:
            if file_name.endswith('.json'):
                json.dump(self._mutant_info, f) # for .json
            elif file_name.endswith('.jsonl'):
                for mut_info in self._mutant_info: # for .jsonl
                    json.dump(mut_info, f)
                    f.write('\n')
            else:
                raise ValueError(f'Unsupported file format in {file_name}, expected .json or .jsonl')

class BCBEvaluator:
    def __init__(self, testset_file, template_file, output_dir):
        self.testset = pd.read_csv(testset_file).loc[49:50]#.loc[:10]
        self._template_file = template_file
        self.output_dir = output_dir
        # super(BCBEvaluator, self).__init__()
    def execute_code(self, code: str, test: str) -> tuple:
        """Execute the code and return status and details"""
        with ProcessPoolExecutor(max_workers=1) as executor:
            kwargs = {
                'code': code,
                'test_code': test,
                'entry_point': None,
                'test_class_name': "TestCases",
                'max_as_limit': 30*1024,
                'max_data_limit': 30*1024,
                'max_stack_limit': 10,
                'min_time_limit': 0.1,
                'gt_time_limit': 2.0,
            }
            future = executor.submit(untrusted_check, **kwargs)
            return future.result()
    def _extract_first_unpass_test(self, buggy_details, test_class_source, dedent_test_method_source=False):
        """
        从buggy_details中提取第一个未通过的test method的信息，包括test method的name，source，以及test_result
        如果buggy_details中有ALL字段，表示整个代码在执行时出现了系统错误，则返回上一轮的test method name和source，并把sys_error的报错信息封装进test_result里
        """
        first_unpass_method_name, first_unpass_method_result = list(buggy_details.items())[0]
        method_source_code = extract_method_source(test_class_source, first_unpass_method_name, dedentation=dedent_test_method_source)

        # 计算method level的pass rate
        all_test_methods = re.findall(r'def\s+(test_\w+)\s*\(', test_class_source)
        pass_count = len(all_test_methods) - len(buggy_details)
        pass_rate = f"{pass_count}/{len(all_test_methods)}"

        return first_unpass_method_name, method_source_code, first_unpass_method_result, pass_rate
    def update_solution_logs(self, coding_task: str, code: str, test: str, status: str, details: dict, step_n: int, solution_version: int, old_focused_test_name, old_focused_test_source, old_pass_rate, logger, output_dir) -> str:
        """Update agent state based on execution results and return observation"""
        # Update the code regardless of execution result
        # solution_version += 1
        solution_code_file = str(output_dir / Path(f"solution_v{solution_version}_s{step_n}_o{int(status=='pass')}.py"))
        
        # Handle successful execution
        if status == 'pass':
            focused_test_name = None
            focused_test_source = None
            focused_test_result = None
            pass_rate = f'{old_pass_rate.split("/")[1]}/{old_pass_rate.split("/")[1]}'
        # Handle system errors or timeouts
        elif status == 'sys_error' or status == 'timeout' or "ALL" in details:
            error_message = details.get('ALL', 'Unknown system error')
            focused_test_name = old_focused_test_name
            focused_test_source = old_focused_test_source
            focused_test_result = {'stat':'sys_error', 'exception_type': error_message, 'stdout_logs': '', 'traceback_frame': [], 'traceback_str': error_message}
            pass_rate = f'0/{old_pass_rate.split("/")[1]}'
        else: # status == 'fail' or 'error'   
            # Extract information about the first failing test
            focused_test_name, focused_test_source, focused_test_result, pass_rate = self._extract_first_unpass_test(details, test)
            if focused_test_name != old_focused_test_name:
                logger.info(f"Focused test method changed from {old_focused_test_name} to {focused_test_name}. Pass Rate changed from {old_pass_rate} to {pass_rate}")
        
        current_solution = {
            "coding_task": coding_task,
            "step_n": step_n,
            "solution_version": solution_version,
            "code_solution": code,
            "code_status": status,
            "focused_test_name": focused_test_name,
            "focused_test_source": focused_test_source,
            "focused_test_result": focused_test_result,
            "pass_rate": pass_rate,
            "solution_code_file": solution_code_file,
        }
        solution_trace_file = str(output_dir / Path(f"solution_trace.csv"))
        pd.DataFrame([current_solution]).to_csv(
            solution_trace_file, 
            mode='a',  # 追加模式
            header=not os.path.isfile(solution_trace_file),  # 仅在文件不存在时写入表头
            index=False
        )

    # def update_solution_code_file(self):
        """ 
        如果action是repair，更新当前的debug code文件，以及focused test信息
        """
        if status == "pass":
            entry_code = "testcases = TestCases()\n"
        entry_code = "testcases = TestCases()\n"
        if "def setUp(self" in test:
            entry_code += "testcases.setUp()\n"
        if status == "pass":
            entry_code += f"# code status: pass"
        else:
            entry_code += f"testcases.{focused_test_name}()\n"
        if "def tearDown(self" in test: # TODO: 这样写的话，如果focused_test_name方法执行时报错了，后面的tearDown()就不会被执行了。这样可能导致一些资源没有被清理，但是这些资源在debug期间保持可被观测 对于debugger来说也是必要的，所以暂时就这样。
            entry_code += "testcases.tearDown()\n"
        # self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version}_s{self.step_n}_o{int(self.code_status=='pass')}.py"))
        open(solution_code_file, 'w').write(code + '\n' + test + '\n' + entry_code)

        return solution_code_file, focused_test_name, focused_test_source, pass_rate

    def get_solutions(self, N=1, task_budget_steps=3, ephemeral_file=EPHEMERAL_FILE):
        if N != 1:
            raise NotImplementedError
        avg_input_token = []
        avg_output_token = []
        avg_model_call = []
        avg_steps = []
        avg_solutions = []
        pass_ids = []
        fail_ids = []
        for idx, problem in tqdm(self.testset.iterrows(), total=len(self.testset)):

            task_id = problem['task_id']
            task_id_num = re.search(r'\d+', problem['task_id']).group()
            task_output_dir = Path(f"{self.output_dir}/{task_id_num}")
            os.makedirs(task_output_dir, exist_ok=True)
            task_logger = setup_separate_log(f"{task_output_dir}/{task_id_num}.log", console=False)
            coding_task = problem['instruct_prompt']
            code_solution = problem['buggy_solution']
            code_status = problem['buggy_status']
            test = problem['test']
            buggy_test_method_exec_results = eval(problem['buggy_details'])
            task_left_steps = task_budget_steps
            task_token_input = 0
            task_token_output = 0
            task_model_call = 0
            solution_version = 0

            ephemeral_file, focused_test_name, focused_test_source, pass_rate = self.update_solution_logs(
                coding_task=coding_task,
                code=code_solution,
                test=test,
                status=code_status,
                details=buggy_test_method_exec_results,
                step_n=0,
                solution_version=solution_version,
                old_focused_test_name=None,
                old_focused_test_source=None,
                old_pass_rate=None,
                logger=task_logger,
                output_dir=task_output_dir
            )
            task_logger.info(f'\n{"="*50}\nStart Debugging on Task {task_id}\n{"="*50}')
            logging.info(f'\n{"="*50}\nStart Debugging on Task {task_id}\n{"="*50}')
            while True:
                try:
                    prompt_builder = PromptBuilder(ephemeral_file, self._template_file, verbose=DEBUG, output_file=f"{task_output_dir}/agent_trace.csv")
                    task_logger.info(f'\n{"-"*50}\nStart Debugging on Task {task_id} Solution {solution_version}@Step{prompt_builder.cur_step}:\n{code_solution}\nStatus: {code_status}\nDetails: {buggy_test_method_exec_results}\n{"-"*50}')
                    prompt_builder.take_steps(n_steps=task_left_steps)
                    task_logger.info(f'Debugging Trace on Solution {solution_version} took {prompt_builder.cur_step} steps: {prompt_builder._prompt}')
                    proposed_sol = prompt_builder.get_solution()
                    solution_version += 1

                    # vaildate the patch
                    status, buggy_test_method_exec_details = self.execute_code(proposed_sol, test)
                    task_logger.info(f'Proposed Solution: {proposed_sol}\nStatus: {status}\nDetails: {buggy_test_method_exec_details}')
                    task_logger.info(f'Debugging on Solution {solution_version} consumped step={prompt_builder.cur_step}, input tokens={prompt_builder.input_token}, output tokens={prompt_builder.output_token}, model calls={prompt_builder.model_call}')

                    # update the solution logs
                    ephemeral_file, focused_test_name, focused_test_source, pass_rate = self.update_solution_logs(
                        coding_task=coding_task,
                        code=proposed_sol,
                        test=test,
                        status=status,
                        details=buggy_test_method_exec_details,
                        step_n=(task_budget_steps - task_left_steps) + prompt_builder.cur_step, # 当前任务总debug步数step_n = 上一轮solution消耗的步数 + 本轮solution消耗的步数
                        solution_version=solution_version,
                        old_focused_test_name=focused_test_name,
                        old_focused_test_source=focused_test_source,
                        old_pass_rate=pass_rate,
                        logger=task_logger,
                        output_dir=task_output_dir
                    )

                    if status == 'pass' or task_left_steps <=  1:
                        task_logger.info(f'{"="*50}\nFinished on task {task_id}. halted: {task_left_steps<=1}, status: {status}. Final solution:\n{proposed_sol}')
                        break
                except Exception as e:
                    import traceback
                    print('Warning@PromptBuilder:', type(e), e, traceback.format_exc())
                    task_logger.error(f'Error on task {task_id} solution {solution_version} at step {prompt_builder.cur_step}: {e}\ntraceback: {traceback.format_exc()}')
                    status = 'sys_error'
                    buggy_test_method_exec_details = {'ALL': repr(e)+'\n'+traceback.format_exc()}
                    break
                finally:
                    task_token_input += prompt_builder.input_token
                    task_token_output += prompt_builder.output_token
                    task_model_call += prompt_builder.model_call
                    # count the steps used
                    task_left_steps -= prompt_builder.cur_step

            task_logger.info(f"{'='*50}Task {task_id} Total comuptation consumption:\ninput tokens={task_token_input}, output tokens={task_token_output}, model calls={task_model_call}")
            logging.info(f"{'='*50}Task {task_id} Total comuptation consumption:\ninput tokens={task_token_input}, output tokens={task_token_output}, model calls={task_model_call}")
            
            avg_input_token.append(task_token_input)
            avg_output_token.append(task_token_output)
            avg_model_call.append(task_model_call)
            avg_steps.append(task_budget_steps - task_left_steps)
            avg_solutions.append(solution_version)
            if status == 'pass':
                pass_ids.append(task_id)
            elif status == 'fail':
                fail_ids.append(task_id)
            # TODO: record the solution and execution results: proposed_sol, status, buggy_test_method_exec_details

            # mutant_instance['result'] = result
            # mutant_instance['samples'] = [proposed_sol]
            # mutant_instance['trace'] = prompt_builder._trace
            # mutant_instance['prompt_at_repair'] = prompt_builder._prompt

            # # evalaute one solution at a time, and save to 
            # evaluation_object = {
            #     'task_id': mutant_instance['task_id'],
            #     'samples': mutant_instance['samples']
            # }
            # failing_tests = self.evaluate_sol(evaluation_object)
            # mutant_instance['passed'] = (len(failing_tests) == 0)
        pd.DataFrame({
            "sample_size": [len(self.testset)],
            "avg_input_token": [sum(avg_input_token)/len(avg_input_token)],
            "avg_output_token": [sum(avg_output_token)/len(avg_output_token)],
            "avg_model_call": [sum(avg_model_call)/len(avg_model_call)],
            "avg_steps": [sum(avg_steps)/len(avg_steps)],
            "avg_solutions": [sum(avg_solutions)/len(avg_solutions)],
            "pass_rate": [len(pass_ids)/len(self.testset)],
            "fail_rate": [len(fail_ids)/len(self.testset)],
            "pass_ids": [pass_ids],
            "fail_ids": [fail_ids],
        }).to_csv(f"{self.output_dir}/eval_results.csv", index=False)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutant_file', type=str, default='/data/wyk/autosd/arhe/arhe_data/arhe_bugs.json')#required=True, 
    parser.add_argument('--template_file', type=str, default='/data/wyk/autosd/arhe/prompts/zero_shot_detailed.txt')
    parser.add_argument('--output_file', type=str, default="test.jsonl")
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=1)

    args = parser.parse_args()

    token_enc = AutoTokenizer.from_pretrained("/data/share/qwen/Qwen2.5-Coder-7B-Instruct/")

    output_dir = "/data/wyk/autosd/arhe/bigcodebench5.new"
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/test-bigcodebench.log"
    setup_log(log_file, console=False)

    llm = QWenLLM('pre-qwen-max-2025-01-25-chat')


    # 先清空args.output_file：如果已经存在，就把源文件后面加一个.bak
    # if os.path.exists(args.output_file):
    #     os.rename(args.output_file, args.output_file+'.bak')

    for r_idx in range(args.repeats):
        # asd_evaluator = ASDEvaluator(
        #     args.mutant_file, args.template_file
        # )
        # asd_evaluator.get_solutions(steps=args.steps)

        args.mutant_file = "/data/wyk/bigcodebench/bigcodebench/results/bigcodebench-debug-dataset/debug_gpt-3.5-turbo-0125.csv"
        bcb_evaluator = BCBEvaluator(
            args.mutant_file, args.template_file, output_dir
        )
        bcb_evaluator.get_solutions(task_budget_steps=args.steps)
        # asd_evaluator.evaluate_all_solutions()
        # if args.output_file is not None:
        #     num_out_file = args.output_file if args.repeats == 1 else args.output_file.split('.')[0]+f'_{r_idx}.jsonl'
        #     asd_evaluator.save_to(num_out_file)
        #     print(f'Iter {r_idx+1}/{args.repeats} saved')
