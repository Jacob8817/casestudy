"""Regression tests for Review 2 Module 3, including connected frozen replay."""
import json
import numpy as np
import nibabel as nib
import pytest
import torch
from scipy.special import logsumexp, softmax
from src.evaluation.metrics import _nll, normalized_entropy
from src.refinement.rescue import (_local_tta_prediction, extract_priority_rois, build_rescue_priority,
    run_validation_rescue)
from src.uncertainty.brain_skull import run_validation_uncertainty

T=1.1064573224712873
WEIGHTS=dict(uncertainty=.45,anatomical_violation=.35,boundary_complexity=.20)


def test_nll_objective_matches_explicit_likelihood():
    logits=np.array([[4.,2.,-1.],[0.,3.,1.],[-2.,1.,4.]])
    labels=np.array([0,2,2]);p=softmax(logits/T,axis=1)
    assert _nll(np.log(T),logits,labels)==pytest.approx(-np.log(p[np.arange(3),labels]).mean(),abs=1e-14)
    entropy=normalized_entropy(p,class_axis=1)
    assert np.allclose(entropy,-np.sum(p*np.log(p),axis=1)/np.log(3),atol=1e-7)
    assert np.array_equal(logits.argmax(1),p.argmax(1))


@pytest.mark.parametrize('axes',[(0,),(1,),(2,),(0,1,2)])
def test_inverse_flips_restore_asymmetric_spatial_predictions(axes):
    class Pointwise(torch.nn.Module):
        def forward(self,x):
            assert not self.training and not torch.is_grad_enabled()
            return torch.cat([x[:,0:1],-x[:,0:1],x[:,1:2]],dim=1),x
    image=np.random.default_rng(701).normal(size=(2,7,8,9)).astype(np.float32)
    expected=np.stack([image[0],-image[0],image[1]]).argmax(0)
    actual=_local_tta_prediction(Pointwise(),image,device=torch.device('cpu'),axes=axes)
    assert np.array_equal(actual,expected)


def test_ensemble_averages_logits_not_hard_votes_or_probabilities():
    class VariantModel(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):
            values=[[10,0,0],[0,4,0],[0,4,0],[0,0,4]][self.calls];self.calls+=1
            return torch.tensor(values,dtype=torch.float32).reshape(1,3,1,1,1).expand(1,3,*x.shape[2:]),x
    model=VariantModel()
    result=_local_tta_prediction(model,np.zeros((2,4,5,6),np.float32),device=torch.device('cpu'),axes=(0,1,2))
    assert model.calls==4 and (result==0).all()
    assert softmax(np.array([[10,0,0],[0,4,0],[0,4,0],[0,0,4]]),axis=1).mean(0).argmax()==1


def test_roi_selection_ranks_components_limits_three_and_respects_physical_volume():
    mask=np.zeros((128,128,128),bool);q=np.zeros(mask.shape,np.float32)
    starts=[(5,5,5),(110,5,5),(5,110,5),(110,110,110)]
    for weight,start in enumerate(starts,1):
        sl=tuple(slice(v,v+5) for v in start);mask[sl]=True;q[sl]=weight/4
    rois=extract_priority_rois(mask,q,(1.5,)*3,roi_size=(96,)*3,minimum_component_ml=.25,maximum_rois=3)
    assert len(rois)==3
    assert [[s.start for s in roi] for roi in rois]==[[32,32,32],[0,32,0],[32,0,0]]
    assert all(tuple(s.stop-s.start for s in roi)==(96,)*3 for roi in rois)
    assert extract_priority_rois(mask,q,(.5,)*3,roi_size=(96,)*3,minimum_component_ml=.25,maximum_rois=3)==[]


def test_q_formula_and_invalid_inputs():
    labels=np.zeros((12,13,14),np.uint8);labels[3:9,3:9,3:9]=2
    u=np.full(labels.shape,.6,np.float32);v=np.full(labels.shape,.2,np.float32)
    q,b=build_rescue_priority(u,v,labels,WEIGHTS)
    assert np.allclose(q,.45*u+.35*v+.20*b,atol=1e-7)
    assert b[3,6,6]>b[0,0,0]
    with pytest.raises(ValueError):build_rescue_priority(u*np.nan,v,labels,WEIGHTS)
    with pytest.raises(ValueError):build_rescue_priority(u,v,labels,dict(uncertainty=-.1,anatomical_violation=.5,boundary_complexity=.6))


def test_connected_frozen_pipeline_never_refits_and_replaces_only_selected(tmp_path,monkeypatch):
    import src.uncertainty.brain_skull as uncertainty
    import src.refinement.rescue as rescue
    def forbidden(*args,**kwargs):raise AssertionError('Frozen replay must not optimize')
    monkeypatch.setattr(uncertainty,'fit_temperature',forbidden)
    monkeypatch.setattr(uncertainty,'select_error_threshold',forbidden)
    monkeypatch.setattr(rescue,'select_error_threshold',forbidden)
    shape=(16,16,16);image=np.zeros((2,*shape),np.float32);labels=np.zeros(shape,np.uint8);labels[3:9,3:9,3:9]=2
    np.save(tmp_path/'image.npy',image);np.save(tmp_path/'labels.npy',labels)
    record=dict(case_id='unit',split='val',shape=list(shape),ct_affine=np.eye(4).tolist(),spacing_mm=[1]*3,
        image_path=str(tmp_path/'image.npy'),label_path=str(tmp_path/'labels.npy'))
    logits=np.zeros((3,*shape),np.float32);logits[0]=4;logits[:,4:12,4:12,4:12]=.1
    monkeypatch.setattr(uncertainty,'infer_multitask_logits',lambda *a,**kw:(logits,np.zeros((2,*shape),np.float32)))
    frozen=dict(temperature_scaling=dict(temperature=T),selected_error_threshold=dict(threshold=.5))
    c4=dict(calibration_samples_per_case=500,random_seed=44,ece_bins=15,error_analysis_samples_per_case=500,
        method='scalar_temperature_scaled_softmax_entropy',interpretation='predictive_not_epistemic')
    p4=run_validation_uncertainty(None,[record],c4,tmp_path/'p4',device=torch.device('cpu'),roi=8,overlap=.25,
        frozen_calibration_report=frozen,progress=lambda _:None)
    qc=tmp_path/'qc/unit';qc.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros(shape,np.float32),np.eye(4)),qc/'anatomical_violation.nii.gz')
    monkeypatch.setattr(rescue,'_local_tta_prediction',lambda model,patch,**kw:np.full(patch.shape[1:],2,np.uint8))
    c5=dict(random_seed=45,threshold_samples_per_case=500,priority_weights=WEIGHTS,roi_size=[8]*3,
        minimum_component_ml=.001,maximum_rois_per_case=3,tta_spatial_axes=[0,1,2],refinement='local_flip_tta')
    p5=run_validation_rescue(None,[record],p4,c5,tmp_path/'qc',tmp_path/'p4',tmp_path/'p5',
        device=torch.device('cpu'),frozen_priority_threshold=dict(threshold=.3),progress=lambda _:None)
    case=p5['cases'][0];assert case['changed_voxels']>0 and case['changed_outside_selection']==0
    assert p4['temperature_refitted'] is False and p5['priority_threshold_reselected'] is False
    changed=np.asarray(nib.load(tmp_path/'p5/cases/unit/changed_voxels.nii.gz').dataobj)>0
    selected=np.asarray(nib.load(tmp_path/'p5/cases/unit/selected_replacement_mask.nii.gz').dataobj)>0
    assert not (changed&~selected).any() and not selected.all()
    shifted=np.eye(4);shifted[0,3]=1
    nib.save(nib.Nifti1Image(np.zeros(shape,np.float32),shifted),qc/'anatomical_violation.nii.gz')
    with pytest.raises(ValueError,match='grid mismatch'):
        run_validation_rescue(None,[record],p4,c5,tmp_path/'qc',tmp_path/'p4',tmp_path/'bad',device=torch.device('cpu'),frozen_priority_threshold=dict(threshold=.3))


def test_validation_entrypoints_reject_locked_test_before_io(tmp_path):
    record=dict(split='test')
    with pytest.raises(ValueError,match='validation records only'):
        run_validation_rescue(None,[record],{}, {},tmp_path,tmp_path,tmp_path,device=torch.device('cpu'))
    with pytest.raises(ValueError,match='validation records only'):
        run_validation_uncertainty(None,[record],{},tmp_path,device=torch.device('cpu'),roi=96,overlap=.25)
