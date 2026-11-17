"""Review 2 Module 3 integrated verification, frozen validation replay only.

No test payloads, optimization, retraining, CASE_001 changes, or original-output writes.
"""
from pathlib import Path
import sys, json, hashlib, gc, csv, datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
import yaml
from scipy.special import softmax
from scipy.ndimage import binary_erosion,binary_dilation,gaussian_filter
from src.training.anatomy_phase2 import build_phase2_model,sha256_file
from src.uncertainty.brain_skull import run_validation_uncertainty
from src.refinement.rescue import run_validation_rescue,_sample_indices
from src.anatomy.violations import anatomy_violation_map
from src.geometry.surface import binary_surface_metrics
from src.evaluation.metrics import _nll

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/module3_verification'
FRAME=ROOT/'outputs/anatomy_aware_framework'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    cfg=yaml.safe_load((ROOT/'configs/anatomy_aware_phases4_10_v1.yaml').read_text())
    model_cfg=yaml.safe_load((ROOT/'configs/anatomy_aware_v1.yaml').read_text())
    qc_cfg=yaml.safe_load((ROOT/'configs/anatomy_aware_phase3_v1.yaml').read_text())['quality_control']
    frozen4=json.loads((FRAME/'phase4/phase4_report.json').read_text());frozen5=json.loads((FRAME/'phase5/phase5_report.json').read_text())
    t=frozen4['temperature_scaling']['temperature'];assert t==1.1064573224712873
    threshold=frozen5['selected_priority_threshold'];assert threshold['threshold']==0.4128813147544861
    manifest=json.loads((ROOT/'data/splits/dataset_manifest.json').read_text())
    splits={c['case_id']:c['split'] for c in manifest['cases']}
    ids=[c['case_id'] for c in frozen4['cases']];assert ids==['s0346','s0792','s0874','s1161','s1260']
    records=[json.loads((ROOT/f'data/cache/anatomy_aware_phase2/{case}/metadata.json').read_text()) for case in ids]
    assert all(r['split']==splits[r['case_id']]=='val' for r in records)
    distance=json.loads((FRAME/'phase3/anatomy_distance_reference.json').read_text())
    assert all(splits[c]=='train' for c in distance['case_ids'])
    checkpoint=FRAME/'phase3/best.pt'
    phase3=json.loads((FRAME/'phase3/phase3_report.json').read_text())
    assert sha256_file(checkpoint)==phase3['model_report']['best_checkpoint_sha256']
    # Guard source caches as well as historical artifacts and CASE_001 exports.
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    for r in records:
        for key in ('image_path','label_path'):
            protected[r[key]]=sha256_file(Path(r[key]))
    dump(OUT/'protected_hashes.json',protected)
    import uuid
    replay=OUT/('replay_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    replay.mkdir()
    dump(OUT/'latest_run.json',dict(path=str(replay)))
    torch.manual_seed(44);np.random.seed(44);torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    model=build_phase2_model(model_cfg).to('cuda')
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    model_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    observations=[]

    def observer(record,logits,sdf,p,u):
        case=record['case_id'];print('Independent U/V verification:',case,flush=True)
        expected=softmax(logits/t,axis=0)
        p_error=float(np.max(np.abs(expected-p)));sum_error=float(np.max(np.abs(p.sum(0)-1)))
        positive=np.maximum(p,np.float32(1e-7))
        u_expected=-np.sum(positive*np.log(positive),axis=0)/np.log(3)
        u_error=float(np.max(np.abs(u_expected-u)))
        assert p_error==0 and sum_error<1e-6 and u_error<1e-6
        old_u=np.asarray(nib.load(FRAME/f'phase4/cases/{case}/predictive_entropy.nii.gz').dataobj)
        old_prediction=np.asarray(nib.load(FRAME/f'phase4/cases/{case}/initial_prediction.nii.gz').dataobj)
        observed=dict(case_id=case,probability_max_error=p_error,probability_sum_max_error=sum_error,entropy_max_error=u_error,
            entropy_vs_original_max_error=float(np.max(np.abs(u-old_u))),initial_prediction_changed_voxels=int(np.count_nonzero(p.argmax(0)!=old_prediction)))
        raw=softmax(logits,axis=0)
        saved_v=np.asarray(nib.load(FRAME/f'phase3/qc_phase3/{case}/anatomical_violation.nii.gz').dataobj)
        weighted=np.zeros(u.shape,np.float32);errors={}
        with np.load(FRAME/f'phase3/qc_phase3/{case}/violation_components.npz') as archive:
            for name,weight in qc_cfg['violation_weights'].items():
                values=archive[name]
                assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
                weighted+=weight*values
                if name=='probability_ambiguity':
                    errors[name]=float(np.max(np.abs(values-np.clip(4*raw[1]*raw[2],0,1))))
                if name=='sdf_sign_disagreement':
                    independent=np.maximum((old_prediction==1)!=(sdf[0]>0),(old_prediction==2)!=(sdf[1]>0))
                    errors[name]=float(np.max(np.abs(values-independent)))
        errors['weighted_V']=float(np.max(np.abs(weighted-saved_v)))
        assert errors['weighted_V']<1e-6 and errors['probability_ambiguity']<2e-6
        assert errors['sdf_sign_disagreement']==0
        if case=='s0346':
            rebuilt,parts,_=anatomy_violation_map(old_prediction,raw,sdf,tuple(record['spacing_mm']),distance,
                minimum_component_ml=qc_cfg['disconnected_minimum_ml'],minimum_largest_fraction=qc_cfg['disconnected_minimum_largest_fraction'],weights=qc_cfg['violation_weights'])
            errors['all_V_components_recomputed_max_error']=float(np.max(np.abs(rebuilt-saved_v)))
            assert errors['all_V_components_recomputed_max_error']<2e-6
        observed['violation_checks']=errors
        observed['violation_probability_basis']='Uncalibrated phase3 softmax, as used to select the frozen Q threshold; U uses calibrated softmax.'
        observations.append(observed);dump(replay/'observations.json',observations)
        # Complete calibrated probability volume, without overwriting phase4 history.
        folder=replay/'phase4/cases'/case;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'calibrated_probabilities.npy',p)
        del expected,old_u,raw,weighted,saved_v
        gc.collect()

    print('Running connected frozen calibration -> U -> Q -> local TTA pipeline',flush=True)
    p4=run_validation_uncertainty(model,records,cfg['phase4_uncertainty'],replay/'phase4',device=torch.device('cuda'),
        roi=model_cfg['inference']['roi_size'][0],overlap=model_cfg['inference']['overlap'],
        frozen_calibration_report=frozen4,artifact_observer=observer,progress=lambda m:print(m,flush=True))
    # Verify a minimum of validation NLL without refitting or choosing any parameter.
    with np.load(replay/'phase4/calibration_samples.npz') as samples:
        logits=samples['logits'].astype(np.float64);labels=samples['labels'].astype(np.int64)
    beta=1/t;p=softmax(beta*logits,axis=1)
    expected_z=np.sum(p*logits,axis=1)
    derivative=float(np.mean(expected_z-logits[np.arange(len(labels)),labels]))
    curvature=float(np.mean(np.sum(p*(logits-expected_z[:,None])**2,axis=1)))
    nll=_nll(np.log(t),logits,labels)
    left=_nll(np.log(t*.99),logits,labels);right=_nll(np.log(t*1.01),logits,labels)
    assert curvature>0 and nll<left and nll<right
    assert abs(derivative)<1e-5
    assert abs(p4['calibration_before']['nll']-frozen4['calibration_before']['nll'])<1e-6
    assert abs(p4['calibration_after']['nll']-frozen4['calibration_after']['nll'])<1e-6
    calibration=dict(temperature=t,nll_before=p4['calibration_before']['nll'],nll_after=nll,
        nll_at_0_99_T=left,nll_at_1_01_T=right,derivative_wrt_inverse_temperature=derivative,curvature_wrt_inverse_temperature=curvature,
        validation_samples=len(labels),refitted=False)
    dump(replay/'calibration_verification.json',calibration)
    p5=run_validation_rescue(model,records,p4,cfg['phase5_rescue'],FRAME/'phase3/qc_phase3',replay/'phase4',replay/'phase5',
        device=torch.device('cuda'),frozen_priority_threshold=threshold,progress=lambda m:print(m,flush=True))
    cases=[];q_samples=[];error_samples=[]
    for record,case_report in zip(records,p5['cases']):
        case=record['case_id'];print('Saved-artifact verification and surface metrics:',case,flush=True)
        a=record['ct_affine'];case_root=replay/'phase5/cases'/case
        load=lambda path:np.asarray(nib.load(path).dataobj)
        initial=load(replay/f'phase4/cases/{case}/initial_prediction.nii.gz');refined=load(case_root/'refined_prediction.nii.gz')
        target=np.load(record['label_path'],mmap_mode='r');q=load(case_root/'rescue_priority.nii.gz')
        u=load(replay/f'phase4/cases/{case}/predictive_entropy.nii.gz');v=load(FRAME/f'phase3/qc_phase3/{case}/anatomical_violation.nii.gz')
        foreground=initial>0;b=foreground^binary_erosion(foreground)
        for cid in (1,2):m=initial==cid;b|=m^binary_erosion(m)
        b=np.clip(gaussian_filter(binary_dilation(b,iterations=2).astype(np.float32),sigma=1),0,1)
        q_error=float(np.max(np.abs(q-(.45*u+.35*v+.20*b))))
        assert q_error<1e-7 and np.array_equal(b,load(case_root/'boundary_signal.nii.gz'))
        candidate=q>=threshold['threshold'];selected=np.zeros(initial.shape,bool)
        for roi in case_report['roi_locations']:
            sl=tuple(slice(*s) for s in roi['slices_start_stop']);selected[sl]|=candidate[sl]
            assert roi['padded_inference_shape']==[96]*3
        assert case_report['rois']<=3
        changed=initial!=refined
        assert not np.any(changed&~selected)
        assert np.array_equal(selected,load(case_root/'selected_replacement_mask.nii.gz')>0)
        old_refined=load(FRAME/f'phase5/cases/{case}/refined_prediction.nii.gz')
        old_q=load(FRAME/f'phase5/cases/{case}/rescue_priority.nii.gz')
        old_difference=int(np.count_nonzero(refined!=old_refined))
        index=_sample_indices(target.size,cfg['phase5_rescue']['threshold_samples_per_case'],45,case)
        q_samples.append(q.ravel()[index]);error_samples.append((initial!=target).ravel()[index])
        surface={}
        for cid,name in [(1,'brain'),(2,'skull')]:
            surface[name]={stage:binary_surface_metrics(prediction==cid,target==cid,tuple(record['spacing_mm'])) for stage,prediction in [('before',initial),('after',refined)]}
        cases.append({**case_report,'priority_formula_max_error':q_error,'priority_vs_original_max_error':float(np.max(np.abs(q-old_q))),
            'refined_vs_original_changed_voxels':old_difference,'surface_metrics':surface})
        dump(replay/'case_verification.json',cases)
        del initial,refined,target,q,u,v,b,old_refined,old_q,candidate,selected,changed
        gc.collect()
    # Verify the historical threshold is still the F1 maximum on its deterministic
    # validation sample. This is an audit, not a new threshold selection or write.
    from sklearn.metrics import precision_recall_curve
    scores=np.concatenate(q_samples);errors=np.concatenate(error_samples)
    precision,recall,thresholds=precision_recall_curve(errors,scores)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    best=int(np.argmax(f1));assert float(thresholds[best])==threshold['threshold']
    assert all(torch.equal(v.cpu(),model_before[k]) for k,v in model.state_dict().items())
    for name,digest in protected.items():assert sha256_file(ROOT/name)==digest,f'Protected artifact changed: {name}'
    summary=dict(status='verified',slides=json.loads((OUT/'slides_evidence.json').read_text()),calibration=calibration,
        priority=dict(weights=cfg['phase5_rescue']['priority_weights'],threshold=threshold,verified_threshold_f1=float(f1[best]),reselected=False),
        cases=cases,observations=observations,means=p5['means'],checkpoint_sha256=sha256_file(checkpoint),
        protected_artifacts_unchanged=True,official_test_payloads_accessed=False,CASE_001_unchanged=True,
        source_files_changed=['src/uncertainty/brain_skull.py','src/refinement/rescue.py','scripts/build_anatomy_phases4_10_notebook.py','18_anatomy_aware_framework_phases4_10.ipynb','scripts/verify_module3.py','tests/test_module3_verification.py'],
        statuses={'NLL':'FIXED','Priority Q(x)':'FIXED','Local Flip Ensembling':'FIXED'},
        scope='Formulas preserved. Fixed frozen replay integration, spatial guards, and executed selection audit; ROI computation optimized without changing ranking.',
        limitations=['Five development-validation cases, also used historically for calibration/threshold selection; these are not independent generalization estimates.',
            'Dataset reference labels are not newly expert-verified patient ground truth.',
            'V ambiguity uses phase3 uncalibrated probabilities; preserved to retain frozen threshold validity.',
            'B is a dilated/Gaussian-smoothed voxel boundary signal, not physical curvature.',
            'TTA averages four logit fields (identity plus three single-axis flips), not all eight flip combinations. Positive scalar T does not change argmax of their mean.',
            'Overlapping ROIs use deterministic later-ROI-wins replacement; candidate count differs from visited and actually changed voxel counts.',
            'Surface distances use existing deterministic caps at 200,000 boundary points, not exact full-surface distances.',
            'roi_margin_voxels=8 is an unused legacy config entry; actual ROIs are fixed 96 cubes centered on weighted components. No margin promised in Module 3.',
            'All anatomical V components recomputed end-to-end on s0346; saved component aggregation, probability ambiguity and SDF sign verified on all five cases.'])
    dump(replay/'verification_report.json',summary);write_markdown(replay,summary)
    print('MODULE 3 VERIFIED:',replay,flush=True)


def write_markdown(path,r):
    lines=['# Review 2 Module 3 verification','',
        'Slides 8 and 10 inspected in CS16_Review2_Ten_References.pptx. The other three CS16 Review2 deck variants have identical Module 3 slide-10 requirements. No deck edits.', '',
        '| Component | Status | Finding |','|---|---|---|',
        '| NLL temperature calibration | FIXED | Existing log-temperature validation-NLL objective is correct. Added frozen replay; no temperature refit. |',
        '| Priority Q(x) | FIXED | Existing 0.45 U + 0.35 V + 0.20 B preserved. Added frozen threshold replay and affine/finite-input checks. |',
        '| Local flip ensembling | FIXED | Existing inverse flips and mean logits verified. Added ROI locations, candidate/visited/changed counts, selection masks and executable invariant. |', '',
        '## Numerical evidence','',f'Calibration: `{r["calibration"]}`.',
        f'Priority threshold: `{r["priority"]["threshold"]["threshold"]}`; verified sample F1: `{r["priority"]["verified_threshold_f1"]}`. No threshold adjustment.', '',
        '| Case | ROIs | Q candidates | Selected in ROIs | Changed | Outside selection | Brain Dice before / after | Skull Dice before / after |','|---|---:|---:|---:|---:|---:|---|---|']
    for c in r['cases']:
        lines.append(f'| {c["case_id"]} | {c["rois"]} | {c["priority_candidate_voxels"]} | {c["selected_replacement_voxels"]} | {c["changed_voxels"]} | {c["changed_outside_selection"]} | {c["before_dice"]["brain"]:.8f} / {c["after_dice"]["brain"]:.8f} | {c["before_dice"]["skull"]:.8f} / {c["after_dice"]["skull"]:.8f} |')
    lines+=['','## ROI locations','', 'Index ranges are half-open [start, stop), in the cached CT grid; JSON also records RAS-mm origins.']
    for c in r['cases']:
        lines+=['',f'**{c["case_id"]}**']+['- '+str(roi['slices_start_stop']) for roi in c['roi_locations']]
    lines+=['','## Segmentation and surface metrics','','| Case | Class | ASSD mm before / after | HD95 mm before / after | IoU before / after |','|---|---|---|---|---|']
    for c in r['cases']:
        for name,m in c['surface_metrics'].items():
            a,b=m['before'],m['after'];lines.append(f'| {c["case_id"]} | {name} | {a["assd_mm"]:.6f} / {b["assd_mm"]:.6f} | {a["hd95_mm"]:.6f} / {b["hd95_mm"]:.6f} | {a["iou"]:.6f} / {b["iou"]:.6f} |')
    lines+=['','## Integration and changes','',
        'Actual frozen checkpoint -> calibrated probabilities/entropy -> saved phase3 anatomical V -> Q -> frozen selection threshold -> local frozen inference -> inverse transforms -> mean logits -> selected-only replacement -> saved segmentation/surface checks was executed on all five cases.',
        'Notebook 18 and its generator now load historical calibration/thresholds and write to a new replay directory. CASE_001 files and all protected original outputs were hash-checked unchanged. No test payloads, retraining or split changes.', '',
        'Files changed:']+['- `'+name+'`' for name in r['source_files_changed']]
    lines+=['','## Limitations','']+['- '+x for x in r['limitations']]
    lines+=['','Full machine-readable evidence: [verification_report.json](verification_report.json). Executed unit/regression results: [tests.xml](../tests.xml).']
    (path/'verification_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
