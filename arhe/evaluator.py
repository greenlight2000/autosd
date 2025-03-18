import signal
import json
import ast

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, handler)

def read_solutions(filename):
    sols = []
    with open(filename) as f:
        for line in f:
            sols.append(json.loads(line))
    return sols

class HEDataObject():
    def __init__(self):
        self._he_data = self._read_humaneval_data()
        self._problem_num = len(self._he_data)
    
    def _read_humaneval_data(self):
        he_data = dict()
        with open('./arhe_data/HumanEval.jsonl') as f:
            for line in f:
                line_obj = json.loads(line)
                he_data[line_obj['task_id']] = line_obj
        return he_data

class Evaluator(HEDataObject):
    def _get_exec_code(self, solution, custom_test=None, wrap_with_f=True):
        sol_task_id = solution['task_id']
        corresp_he_data = self._he_data[sol_task_id]
        
        llm_prompt = corresp_he_data['prompt']
        sample = solution['samples'][0].strip() + '\n'
        task_name = llm_prompt.split('def ')[-1].split('(')[0]
        llm_prompt = 'def '.join(llm_prompt.split('def ')[:-1])
        
        if custom_test is None:
            code_to_run = llm_prompt + sample + corresp_he_data['test']
        else:
            code_to_run = llm_prompt + sample + custom_test
        if wrap_with_f:
            code_to_run += f'\n\ncheck({task_name})'
            code_to_run = '\n'.join('    '+l for l in code_to_run.split('\n'))
            code_to_run = 'def go():\n' + code_to_run + '\ngo()'
        else:
            assert custom_test is not None
            l = custom_test.replace('candidate', task_name).strip()
            if '==' in l:
                separator = '=='
            elif '<' in l:
                separator = '<'
            elif '>' in l:
                separator = '>'
            elif ' is ' in l:
                separator = ' is '
            else:
                print('Separator unknown for', l)
                raise ValueError('Expected value extraction failed for assertion `'+l+'`')
            
            actual_value = l.split(separator)[0][6:].strip()

            r = ast.parse(l)
            r.body[0].msg = ''
            raw_assert = ast.unparse(r)
            custom_test_modified = raw_assert + ', ' + actual_value
            
            code_to_run = llm_prompt + sample + custom_test_modified
            
        return code_to_run
    # for bigcodebench
    # def _get_exec_code(self, solution, custom_test=None, wrap_with_f=True):
    #     """ 
    #     如果action是repair，更新当前的debug code文件，以及focused test信息
    #     """
    #     if self.code_status == "pass":
    #         entry_code = "testcases = TestCases()\n"
    #     entry_code = "testcases = TestCases()\n"
    #     if "def setUp(self" in self.test:
    #         entry_code += "testcases.setUp()\n"
    #     if self.code_status == "pass":
    #         entry_code += f"# code status: pass"
    #     else:
    #         entry_code += f"testcases.{self.focused_test_name}()\n"
    #     if "def tearDown(self" in self.test: # TODO: 这样写的话，如果focused_test_name方法执行时报错了，后面的tearDown()就不会被执行了。这样可能导致一些资源没有被清理，但是这些资源在debug期间保持可被观测 对于debugger来说也是必要的，所以暂时就这样。
    #         entry_code += "testcases.tearDown()\n"
    #     self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version}_s{self.step_n}_o{int(self.code_status=='pass')}.py"))
    #     open(self.solution_code_file, 'w').write(self.code_solution + '\n' + self.test + '\n' + entry_code)
    #     return llm_prompt + sample + BCB_testclass + entry_code
    
    def evaluate_sol(self, solution):
        task_id = solution['task_id']
        failing_tests = []
        for isolated_test_func in self._isolated_test_generator(task_id):
            code_to_run = self._get_exec_code(solution, isolated_test_func)
        
            try:
                signal.alarm(1)
                exec(code_to_run)
            except Exception as e:
                failing_test = isolated_test_func.split('\n')[-1]
                failing_tests.append({
                    'failing_assertion': failing_test,
                    'failing_exception': str(type(e)),
                })
            finally:
                signal.alarm(0)
                
        return failing_tests
    # TODO: for bigcodebench
    # def execute_code(self, code: str) -> tuple:
    #     """Execute the code and return status and details"""
    #     with ProcessPoolExecutor(max_workers=1) as executor:
    #         kwargs = {
    #             'code': code,
    #             'test_code': self.test,
    #             'entry_point': self.entry_point,
    #             'test_class_name': "TestCases",
    #             'max_as_limit': 30*1024,
    #             'max_data_limit': 30*1024,
    #             'max_stack_limit': 10,
    #             'min_time_limit': 0.1,
    #             'gt_time_limit': 2.0,
    #         }
    #         future = executor.submit(untrusted_check, **kwargs)
    #         return future.result()
    
    def _isolated_test_generator(self, task_id):
        org_test_func = self._he_data[task_id]['test']
        org_test_func = 'def ' + org_test_func.split('def ')[-1]
        parsed_test_func = ast.parse(org_test_func)
        for statement in parsed_test_func.body[0].body:
            if not isinstance(statement, ast.Assert):
                if ast.unparse(statement).startswith('print'):
                    continue
                raise ValueError(f'Unknown statement {ast.unparse(statement)} in {task_id}')
            new_func_ast = ast.parse(org_test_func)
            new_func_ast.body[0].body = [statement]
            yield ast.unparse(new_func_ast)
