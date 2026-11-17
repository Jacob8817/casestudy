"""Frozen external-case inference only; no training, calibration fitting or test access."""
from pathlib import Path
import sys, json, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import torch
import yaml
from scipy.special import softmax
from src.training.anatomy_phase2 import build_phase2_model, sha256_file
from src.uncertainty.brain_skull import infer_multitask_logits
from src.evaluation.metrics import normalized_entropy


def run(output: Path):
    root = Path(__file__).resolve().parents[1]
    source = root/'outputs/10_totalsegmentator_sharp_skull/input/CASE_001_bone_kernel_ct.nii.gz'
    checkpoint = root/'outputs/anatomy_aware_framework/phase3/best.pt'
    calibration = root/'outputs/anatomy_aware_framework/phase4/phase4_report.json'
    locked = json.loads((root/'outputs/anatomy_aware_framework/phase3/phase3_report.json').read_text())
    assert sha256_file(checkpoint) == locked['model_report']['best_checkpoint_sha256']
    contract = dict(case_id='CASE_001', source_ct_sha256=sha256_file(source), checkpoint_sha256=sha256_file(checkpoint), calibration_sha256=sha256_file(calibration), spacing_mm=[1.5]*3, orientation='RAS', interpolation_order=1, seed=2026)
    output.mkdir(parents=True, exist_ok=True)
    meta = output/'confidence_provenance.json'
    if meta.exists():
        prior = json.loads(meta.read_text())
        assert prior['contract'] == contract, 'Stale confidence cache'
        for name, digest in prior['output_hashes'].items():
            assert sha256_file(output/name) == digest
        return prior
    torch.manual_seed(2026); np.random.seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = yaml.safe_load((root/'configs/anatomy_aware_v1.yaml').read_text())
    image = nib.load(source)
    # Reproducible physical resampling, independent of labels, retains the entire CT FOV.
    model_ct = resample_to_output(image, voxel_sizes=(1.5,)*3, order=1, cval=-1024)
    hu = np.asarray(model_ct.dataobj, dtype=np.float32)
    channels = np.stack([np.clip((hu+100)/300,0,1), np.clip((hu+300)/1800,0,1)])
    if not torch.cuda.is_available():
        raise RuntimeError('Existing model inference requires the configured CUDA environment')
    model = build_phase2_model(cfg).to('cuda')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state'])
    print('Frozen CASE_001 inference:', hu.shape, flush=True)
    logits, _ = infer_multitask_logits(model, channels, device=torch.device('cuda'), roi=cfg['inference']['roi_size'][0], overlap=cfg['inference']['overlap'])
    temperature = json.loads(calibration.read_text())['temperature_scaling']['temperature']
    probabilities = softmax(logits/temperature, axis=0).astype(np.float32)
    entropy = normalized_entropy(probabilities, class_axis=0)
    volumes = {'model_ct':hu, 'predictive_entropy':entropy, 'skull_probability':probabilities[2], 'prediction':probabilities.argmax(axis=0).astype(np.uint8)}
    for name, values in volumes.items():
        nib.save(nib.Nifti1Image(values, model_ct.affine), output/f'{name}.nii.gz')
    np.savez_compressed(output/'logits_probabilities.npz', logits=logits, probabilities=probabilities)
    report = dict(contract=contract, case_id='CASE_001', source_ct_sha256=contract['source_ct_sha256'], temperature=temperature,
        model_affine=model_ct.affine.tolist(), model_shape=list(hu.shape), classes={'background':0,'brain':1,'skull':2},
        source_ct=str(source), checkpoint=str(checkpoint), calibration=str(calibration),
        confidence_definition='P(skull|CT) = softmax(logits / fixed_temperature)[2]',
        uncertainty_definition='H = -sum(p log p)/log(3); predictive uncertainty, not epistemic',
        limitation='External case; calibration transfer is unverified. MONAI probability is not TotalSegmentator confidence. No separate mandible/teeth probability exists.',
        official_test_accessed=False, calibration_refitted=False, model_retrained=False,
        output_hashes={p.name:sha256_file(p) for p in output.iterdir() if p.suffix in ('.gz','.npz')})
    meta.write_text(json.dumps(report, indent=2))
    print('Patient-specific probabilities saved', flush=True)
    return report

if __name__ == '__main__':
    run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/patient_skull_verification/confidence'))
