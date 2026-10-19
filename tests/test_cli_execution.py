import io
import json
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from osworld_agent.actions import Action, validate_action, ACTION_REGISTRY
from osworld_agent.cli_executor import execute_cli, cli_capability, file_check_samples
from osworld_agent.decision_context import action_scope, scoped_system
from osworld_agent.decision_modes import decision_mode
from osworld_agent.env import Observation
from osworld_agent.execution_queue import normalize_verification
from osworld_agent.step_verifier import code_review, unresolved_desktop_checks


def local(tmp_path):
    return SimpleNamespace(cli_target='local', workspace_dir=str(tmp_path), os_name='linux')


def call(env, operation, **kw):
    return execute_cli(env, dict(operation=operation, **kw))


def test_real_file_workflow_preserves_content_and_identity(tmp_path):
    env = local(tmp_path)
    assert call(env, 'mkdir', path='报告')['effect_verified']
    written = call(env, 'write', path='报告/初稿.md', content='# 课程分析\n网页开发适合入门。')
    ino = (tmp_path/'报告/初稿.md').stat().st_ino
    assert written['effect_verified']
    assert written['content_preview'] == '# 课程分析\n网页开发适合入门。'
    assert written['content_truncated'] is False
    assert call(env, 'write', path='报告/初稿.md', content='不能覆盖')['status'] == 'failed'
    copied = call(env, 'copy', path='报告/初稿.md', destination='备份.md')
    moved = call(env, 'rename', path='报告/初稿.md', destination='报告/data.md')
    assert moved['status'] == 'executed' and moved['sha256'] == copied['sha256']
    assert (tmp_path/'报告/data.md').stat().st_ino == ino
    assert not call(env, 'stat', path='报告/初稿.md')['file']['exists']
    assert '网页开发' in call(env, 'read', path='报告/data.md')['stdout']
    assert len(call(env, 'list', path='.')['entries']) == 2
    assert call(env, 'move', path='备份.md', destination='报告/data.md')['status'] == 'failed'
    assert (tmp_path/'备份.md').exists()
    assert call(env, 'move', path='备份.md', destination='报告/备份.md')['effect_verified']


def test_shell_exit_stderr_timeout_and_bounded_output(tmp_path):
    env = local(tmp_path)
    ok = call(env, 'shell', command="printf 'hello'; pwd")
    assert ok['exit_code'] == 0 and str(tmp_path) in ok['stdout']
    bad = call(env, 'shell', command="printf 'problem' >&2; exit 7")
    assert bad['status'] == 'failed' and bad['exit_code'] == 7 and bad['stderr'] == 'problem'
    start = time.monotonic()
    timed = call(env, 'shell', command='sleep 15', timeout_s=.1)
    assert timed['error_code'] == 'cli_timeout' and time.monotonic() - start < 3
    large = call(env, 'shell', command='head -c 50000 /dev/zero')
    assert large['stdout_truncated'] and len(large['stdout']) == 16000
    assert call(env, 'shell', command='pwd', cwd='missing')['status'] == 'failed'


@pytest.mark.parametrize('args', [
    {'operation':'delete','path':'x'}, {'operation':'shell','command':''},
    {'operation':'rename','path':'x'}, {'operation':'write','path':'a','content':3},
    {'operation':'shell','command':'pwd','timeout_s':float('nan')},
    {'operation':'read','path':'x','cwd':3}, {'operation':'stat','path':'x','overwrite':'true'},
    {'operation':'shell','command':'pwd','x':1},
])
def test_invalid_cli_rejected(args):
    assert validate_action(Action('cli', args))


def test_no_implicit_host_fallback_and_guest_round_trip(tmp_path):
    class Controller:
        def run_python_script(self, code):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code, {})
            return {'output': output.getvalue()}
    env = SimpleNamespace(cli_target='guest', controller=Controller(), guest_workspace_dir=str(tmp_path/'guest'),
                          initial_state={'workspace_dir':'/wrong/controller/host'}, os_name='linux')
    assert cli_capability(env)['workspace'] == str(tmp_path/'guest')
    assert call(env,'write',path='guest.md',content='guest')['effect_verified']
    assert (tmp_path/'guest/guest.md').exists()
    env.controller = None
    env.run_python = lambda code: pytest.fail('must not fall back to host')
    assert call(env, 'shell', command='pwd')['status'] == 'rejected'
    env = SimpleNamespace(workspace_dir=str(tmp_path),run_command=env.run_python)
    assert not cli_capability(env)['available']


def test_cli_mode_no_image_needed_and_browser_can_switch(tmp_path):
    cap = cli_capability(local(tmp_path))
    obs = Observation(info={'cli_capability':cap})
    step = {'goal':'保存报告','actions':['cli']}
    profile = decision_mode(obs,step)
    assert profile.mode == 'cli'
    assert 'cli' in action_scope(step,obs,profile=profile)
    prompt = scoped_system(action_scope(step,obs,profile=profile), obs,profile=profile)
    assert 'CLI' in prompt and str(tmp_path) in prompt and 'destination' in prompt
    assert 'cli' in ACTION_REGISTRY
    obs.info['cli_capability']['available'] = False
    assert decision_mode(obs,step).mode != 'cli'
    assert 'cli' not in action_scope(step,obs)


def test_independent_filesystem_review_not_stdout_claim(tmp_path):
    env = local(tmp_path)
    predicates = normalize_verification([{'kind':'file_exists','path':'result.md'},
                                         {'kind':'file_content','path':'result.md','text':'真实结果'}])
    obs = Observation(info={'cli_checks':file_check_samples(env,predicates,lambda:10)})
    report = code_review({'verification':predicates},obs,{'status':'executed','stdout':'成功完成'}, {})
    assert report['status'] == 'failed' and unresolved_desktop_checks(report)
    call(env,'write',path='result.md',content='真实结果')
    obs.info['cli_checks'] = file_check_samples(env,predicates,lambda:10)
    assert code_review({'verification':predicates},obs,{}, {})['status'] == 'passed'
    obs.info['cli_checks'] = []
    assert unresolved_desktop_checks(code_review({'verification':predicates},obs,{}, {}))


def test_jev_cli_four_questions_use_system_evidence_without_fake_screenshot(tmp_path):
    from osworld_agent.model.decision_judge import DecisionJudge
    from test_decision_judge import transport, CONFIG
    requests = []
    judge = DecisionJudge(CONFIG, transport(requests))
    result = call(local(tmp_path), 'write', path='data.txt', content='真实系统文件')
    execution = {**result, 'action':'cli', 'args':{'operation':'write','path':'data.txt'}, 'cli_result':result}
    obs = Observation(info={'cli_result':result})
    out = judge.assess_transition('保存文件',{'goal':'保存文件'},Observation(),obs,execution,{},remaining_s=5)
    assert out['status'] == 'answered' and len(requests[-1]['questions']) == 4
    assert not requests[-1].get('images')
    assert '真实系统文件' not in json.loads(requests[-1]['state'])['execution'].get('stdout','')  # no fabricated output
    reviewed = judge.review('保存文件',{'goal':'保存文件'},Observation(),obs,execution,{},remaining_s=5)
    assert reviewed['status'] == 'passed' and 'cli_result' in requests[-1]['state']


def test_browser_download_identity_survives_cli_move_to_subdirectory(tmp_path):
    from osworld_agent.browser_session import BrowserSession, BrowserOptions
    session = BrowserSession(BrowserOptions(),downloads_dir=str(tmp_path))
    source=tmp_path/'paper.pdf';source.write_bytes(b'%PDF-1.7\n real test bytes')
    artifact=session._record_artifact(path=str(source),url='https://arxiv.org/pdf/1234.56789',
                                     final_url='https://arxiv.org/pdf/1234.56789',download_id='one')
    (tmp_path/'reports').mkdir()
    result=call(local(tmp_path),'rename',path='paper.pdf',destination='reports/data.pdf')
    assert session.record_cli_relocation(result)
    saved=session._artifacts[0]
    assert saved['artifact_id']==artifact['artifact_id'] and saved['initial_sha256']==artifact['sha256']
    assert saved['path']==str(tmp_path/'reports/data.pdf')
    assert saved['rename_history'][-1]['source']=='verified_cli_move'
    assert not session.record_cli_relocation({**result,'path':str(tmp_path/'invented.pdf')})
    session.close()


def test_pipeline_routes_cli_without_grounding_and_retains_receipt(tmp_path):
    from osworld_agent.pipeline import Pipeline, StepState
    from osworld_agent.pipeline_format import _operation_mode
    from test_pipeline_completion import Env, Model
    env=Env();env.cli_target='local';env.workspace_dir=str(tmp_path)
    pipe=Pipeline(model=Model([]),env=env,grounding=None,system_prompt='test')
    state=StepState(task='写文件',obs=env.obs)
    action=Action('cli',{'operation':'write','path':'report.txt','content':'real'})
    resolved=pipe.ground(action,state)
    assert state.route=={'channel':'cli'} and _operation_mode(resolved,state.route)=='cli'
    execution=pipe.execute(resolved,state)
    assert execution.kind=='cli' and execution.info['effect_verified']
    pipe._update_program_info(resolved,execution,state)
    assert state.execution_feedback['cli_result']['file']['exists']
    assert (tmp_path/'report.txt').read_text()=='real' and env.actions==[]


def test_files_equal_reads_entire_files_and_detects_late_differences(tmp_path):
    env=local(tmp_path)
    (tmp_path/'a').write_text('x'*20000+'a');(tmp_path/'b').write_text('x'*20000+'b')
    predicate=normalize_verification([{'kind':'files_equal','path':'a','other_path':'b'}])
    obs=Observation(info={'cli_checks':file_check_samples(env,predicate,lambda:10)})
    assert code_review({'verification':predicate},obs,{}, {})['status']=='failed'
    (tmp_path/'b').write_bytes((tmp_path/'a').read_bytes())
    obs.info['cli_checks']=file_check_samples(env,predicate,lambda:10)
    report=code_review({'verification':predicate},obs,{}, {})
    assert report['status']=='passed' and 'other_sha256' in report['evidence'][0]


def test_cli_only_environment_has_no_browser_actions(tmp_path):
    from osworld_agent.adapters.cli_env import CLIEnvironment
    env=CLIEnvironment(workspace_dir=tmp_path)
    env.prepare_workspace();obs=env.reset('保存文件')
    profile=decision_mode(obs)
    assert profile.mode=='cli' and obs.screenshot is None
    assert set(action_scope(None,obs,profile=profile))=={'cli','wait','request_finish','fail'}
    assert env.step(Action('click',{'target':'button'})).info['dispatched'] is False
    assert cli_capability(env)['workspace']==env.initial_state['workspace_dir']


def test_jev_can_choose_cli_from_browser_when_target_supports_it(tmp_path):
    from test_mode_prompts import browser_observation
    obs=browser_observation();obs.info['cli_capability']=cli_capability(local(tmp_path))
    assert decision_mode(obs,judgment={'label':'cli'}).mode=='cli'
    obs.info['cli_capability']['available']=False
    assert decision_mode(obs,judgment={'label':'cli'}).mode!='cli'


def test_literal_file_content_is_not_an_extra_report_request():
    from osworld_agent.report_delivery import wants_report
    assert not wants_report('在 a.txt 写入「课程比较：数据分析800元」，复制为 b.txt')
    assert wants_report('比较两个课程，然后在 a.txt 写入「课程比较」')
