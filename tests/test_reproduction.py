"""Experiment reproducibility and public CLI regression tests."""
from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments_core import runner
from experiments_core.config import load_config, resolve_config
from experiments_core.data import SyntheticGenerator
from experiments_core.reproducibility import build_manifest, run_or_resume
from experiments_core.validation import parse_grid, target_only, value_tag
from scripts import run_subopt_combined, run_subopt_noise_sweep, run_reg_strength_sweep


def raw_config():
    cfg = load_config(ROOT / 'configs/smoke_linear.yaml')
    cfg['experiment'].update(horizon=12, seeds=[0, 1], progress_every=0)
    cfg['dataset'].update(action_dim=4, context_dim=2, m_select=2)
    return cfg


def resolved(raw=None, **kwargs):
    return resolve_config(raw or raw_config(), None, None, None, **kwargs)


def test_feedback_is_shared_and_independent_of_algorithm_order(tmp_path):
    cfg = resolved(cli_subopt_prob=.5, cli_subopt_mode='one_swap')
    original = runner.build_learners
    _, forward, _ = runner.run_experiment(cfg, tmp_path / 'forward')
    with mock.patch.object(runner, 'build_learners', side_effect=lambda c: list(reversed(original(c)))):
        _, reverse, _ = runner.run_experiment(cfg, tmp_path / 'reverse')
    columns = ['seed', 'algorithm', 'final_regret', 'cumulative_suboptimality', 'projection_events']
    pd.testing.assert_frame_equal(forward[columns].sort_values(['seed','algorithm']).reset_index(drop=True),
                                  reverse[columns].sort_values(['seed','algorithm']).reset_index(drop=True))
    assert (forward.groupby('seed').cumulative_suboptimality.nunique() == 1).all()
    # Resetting each learner's stream preserves the existing separate-target experiment.
    cfg_one = resolved(target_only(raw_config(), 'corectron'), cli_subopt_prob=.5, cli_subopt_mode='one_swap')
    _, one, _ = runner.run_experiment(cfg_one, tmp_path / 'one')
    pd.testing.assert_frame_equal(forward.loc[forward.algorithm=='corectron', columns].reset_index(drop=True), one[columns])


@pytest.mark.parametrize('section,key,value', [
    ('experiment','seeds',[]), ('experiment','seeds',[0,0]), ('experiment','seeds',[-1]),
    ('experiment','seeds',[True]), ('experiment','horizon',1.5), ('experiment','horizon',True),
    ('dataset','context_dim',0), ('dataset','action_dim',3.5),
    ('normalization','target_context_l2',float('nan')),
    ('normalization','target_action_residual_l2',float('inf')),
    ('runtime','save_round_metrics','false'),
])
def test_invalid_configuration_rejected(section,key,value):
    raw=raw_config(); raw[section][key]=value
    with pytest.raises(ValueError): resolved(raw)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,0,True])
def test_invalid_cli_regularization_rejected(value):
    with pytest.raises(ValueError): resolved(cli_corectron_lambda=value)


def test_all_learners_disabled_rejected():
    raw=raw_config()
    for learner in raw['learners'].values(): learner['enabled']=False
    with pytest.raises(ValueError, match='At least one'): resolved(raw)


def test_nonunit_norm_bound_metadata_is_correct():
    raw=raw_config(); raw['normalization']['target_context_l2']=2.; raw['dataset']['linear']['true_weight_norm']=3.
    cfg=resolved(raw)
    assert cfg['learners']['corectron']['b_payoff']==6.
    assert cfg['learners']['ons']['true_comparator_in_domain'] is False
    kernel=load_config(ROOT/'configs/main_kernel_t1000_s10.yaml')
    kernel['normalization']['target_context_l2']=2.
    kernel['dataset']['kernel'].update(kernel_type='poly',degree=2,true_weight_norm=3.)
    for key in ['kons','corectron_rbf']: kernel['learners'][key]['kappa']=4.
    cfg=resolved(kernel)
    assert cfg['learners']['kons']['kappa']==4.
    assert cfg['learners']['corectron_rbf']['b_payoff']==12.
    assert cfg['learners']['kons']['true_comparator_in_domain'] is False


def test_zero_noise_and_uniform_override_take_precedence():
    raw=raw_config(); raw['user_policy']={'subopt_noise_std':.7,'subopt_mode':'one_swap','subopt_prob':.8}
    cfg=run_subopt_combined._resolve_score_perturb(raw,'corectron',.001,0.,None)
    assert cfg['user_policy']['subopt_noise_std']==0.
    cfg=run_subopt_noise_sweep._resolve_with_best_c(raw,'corectron',.001,1.,0.,None)
    assert cfg['user_policy']['subopt_noise_std']==0.
    cfg=resolved(raw,cli_subopt_prob=0.,cli_subopt_mode='uniform')
    assert cfg['user_policy']['subopt_prob']==0. and cfg['user_policy']['subopt_mode']=='uniform'


def test_target_only_disables_kernel_algorithms():
    cfg=load_config(ROOT/'configs/main_kernel_t1000_s10.yaml')
    cfg=run_subopt_combined._target_only(cfg,'ons')
    assert [k for k,v in cfg['learners'].items() if v['enabled']]==['ons']


@pytest.mark.parametrize('grid',['','nan','inf','-1'])
def test_grids_validate_finite_nonnegative_values(grid):
    with pytest.raises(ValueError): parse_grid(grid,'grid')


def test_tags_do_not_alias_nearby_values():
    assert value_tag(.12344)!=value_tag(.12345)
    assert value_tag(1.000000001)!=value_tag(1.000000002)
    assert run_reg_strength_sweep._coeff_tag(.1)=='0p1'


def test_exact_resume_and_modified_config_or_output_refusal(tmp_path):
    cfg=resolved(); out=tmp_path/'run'
    first=run_or_resume(cfg,out)
    with mock.patch.object(runner,'run_experiment',side_effect=AssertionError('should reuse')):
        second=run_or_resume(cfg,out)
    np.testing.assert_allclose(first[1].final_regret,second[1].final_regret)
    manifest=json.loads((out/'manifest.json').read_text())
    assert 'experiments_core/action_sets/m_out_of_n.py' in manifest['code_sha256']
    assert 'experiments_core/learners/kernel_cholesky_cache.py' in manifest['code_sha256']
    assert 'scipy' in manifest['environment']['packages']
    changed=copy.deepcopy(cfg); changed['experiment']['horizon']+=1
    with pytest.raises(ValueError,match='--force'): run_or_resume(changed,out)
    (out/'seed_summary.csv').write_text('changed')
    with pytest.raises(ValueError,match='--force'): run_or_resume(cfg,out)


def test_environment_change_and_missing_metadata_refuse_resume(tmp_path):
    cfg=resolved(); out=tmp_path/'run'; run_or_resume(cfg,out)
    with mock.patch.dict(os.environ,{'OMP_NUM_THREADS':'73'}):
        with pytest.raises(ValueError,match='--force'): run_or_resume(cfg,out)
    (out/'manifest.json').unlink()
    with pytest.raises(ValueError,match='--force'): run_or_resume(cfg,out)


def test_failure_cannot_leave_success_or_old_result_files(tmp_path):
    cfg=resolved(); out=tmp_path/'run'; run_or_resume(cfg,out)
    with mock.patch.object(runner,'run_experiment',side_effect=RuntimeError('failure')):
        with pytest.raises(RuntimeError): run_or_resume(cfg,out,force=True)
    assert (out/'FAILED').exists()
    for name in ['SUCCESS','RUNNING','seed_summary.csv','aggregate_summary.csv','round_metrics.csv','manifest.json']:
        assert not (out/name).exists()
    with pytest.raises(ValueError,match='--force'): run_or_resume(cfg,out)
    run_or_resume(cfg,out,force=True)
    assert (out/'SUCCESS').exists() and not (out/'FAILED').exists()


def test_one_seed_has_no_confidence_interval(tmp_path):
    raw=raw_config();raw['experiment']['seeds']=[0]
    _,_,agg=runner.run_experiment(resolved(raw),tmp_path)
    assert agg.final_regret_ci95.isna().all()
    assert agg.total_runtime_ci95.isna().all()


def test_no_round_metrics_sweep_and_plot(tmp_path):
    raw=raw_config();raw['runtime']['save_round_metrics']=False
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(raw))
    env={**os.environ,'MPLCONFIGDIR':str(tmp_path/'mpl')}
    subprocess.run([sys.executable,str(ROOT/'scripts/run_reg_strength_sweep.py'), '--config',str(config),
                    '--targets','corectron','--coeff-grid','1','--output-root',str(tmp_path/'results'),
                    '--figure-root',str(tmp_path/'figures')],check=True,capture_output=True,env=env)
    out=tmp_path/'results/corectron_c1'
    assert (out/'SUCCESS').exists() and not (out/'round_metrics.csv').exists()
    subprocess.run([sys.executable,str(ROOT/'plot_results.py'),'--input',str(out),'--output',str(tmp_path/'plots')],
                   check=True,capture_output=True,env=env)
    assert (tmp_path/'plots/total_runtime.png').exists()


def test_rkhs_norm_does_not_hide_invalid_gram_matrix():
    cfg=resolved(load_config(ROOT/'configs/main_kernel_t1000_s10.yaml'))
    generator=SyntheticGenerator(cfg,0)
    with mock.patch.object(generator,'_kernel_matrix',return_value=-np.eye(16)):
        with pytest.raises(FloatingPointError): generator._rkhs_norm(np.ones((16,10)))
    with mock.patch.object(generator,'_kernel_matrix',return_value=np.full((16,16),np.nan)):
        with pytest.raises(FloatingPointError): generator._rkhs_norm(np.ones((16,10)))


def test_source_changes_within_process_are_rejected():
    from experiments_core import reproducibility
    with mock.patch.object(reproducibility, "code_hashes", return_value={"changed.py": "new"}):
        with pytest.raises(RuntimeError, match="restart"):
            reproducibility.build_manifest(resolved())


def test_source_change_during_run_cannot_mark_success(tmp_path):
    from experiments_core import reproducibility
    actual = runner.run_experiment
    original_hashes = reproducibility.code_hashes
    def run_then_change(cfg, out):
        result = actual(cfg, out)
        reproducibility.code_hashes = lambda: {"changed.py": "new"}
        return result
    try:
        with mock.patch.object(runner, "run_experiment", side_effect=run_then_change):
            with pytest.raises(RuntimeError, match="during the run"):
                run_or_resume(resolved(), tmp_path / "run")
    finally:
        reproducibility.code_hashes = original_hashes
    assert not (tmp_path / "run/SUCCESS").exists()
    assert (tmp_path / "run/FAILED").exists()


def test_degree_one_polynomial_comparator_flags():
    raw=raw_config();raw['dataset']['kernel']={'kernel_type':'poly','degree':1}
    cfg=resolved(raw)
    assert cfg['learners']['ons']['true_comparator_in_domain'] is True
    assert cfg['learners']['kons']['true_comparator_in_domain'] is True
    kernel=load_config(ROOT/'configs/main_kernel_t1000_s10.yaml')
    kernel['dataset']['kernel'].update(kernel_type='poly',degree=1)
    cfg=resolved(kernel)
    assert cfg['learners']['ons']['true_comparator_in_domain'] is True
    assert cfg['learners']['kons']['true_comparator_in_domain'] is True
