import nibabel as nib
import numpy as np
import pytest
from src.geometry.patient_surface import (anatomical_surface, sample_world, confidence_colors,
    color_surface, require_same_grid, require_same_patient, geometry_digest)


def test_affine_sampling_handles_rotation_reflection_translation_and_anisotropy():
    grid = np.indices((6,7,8)).astype(float)
    volume = grid[0] + 2*grid[1] + 3*grid[2]
    affine = np.array([[0,-2,0,31],[.5,0,0,-9],[0,0,3,72],[0,0,0,1]])
    ijk = np.array([[2.3,3.1,4.2],[0,0,0],[5,6,7]])
    world = nib.affines.apply_affine(affine, ijk)
    assert np.allclose(sample_world(volume, world, affine), ijk@[1,2,3])
    assert np.allclose(nib.affines.apply_affine(np.linalg.inv(affine), world), ijk)


def test_invalid_and_outside_samples_are_grey_not_boundary_confidence():
    v = np.ones((4,4,4), dtype=float)
    assert np.isnan(sample_world(v, np.array([[-.1,1,1],[4,1,1]]), np.eye(4))).all()
    assert (confidence_colors(np.array([np.nan,-1,2])) == [150,150,150,255]).all()


def test_low_confidence_preserves_disconnected_anatomy_and_opening():
    mask = np.zeros((18,18,18), dtype=bool)
    mask[3:13,3:13,3:13] = True
    mask[6:10,6:10,:] = False
    mask[15:17,5:9,5:9] = True  # separate anatomical component
    mesh = anatomical_surface(mask, np.eye(4))
    provenance = dict(case_id='unit_fixture', source_ct_sha256='fixture')
    colored, values = color_surface(mesh, np.zeros(mask.shape), np.eye(4), reference_provenance=provenance, confidence_provenance=provenance)
    assert geometry_digest(mesh) == geometry_digest(colored)
    assert len(mesh.split(only_watertight=False)) == 2
    assert (values == 0).all()
    assert (colored.visual.vertex_colors == [255,0,0,255]).all()


def test_fov_boundary_is_not_silently_capped():
    mask = np.zeros((8,8,8), dtype=bool); mask[2:6,2:6,:4] = True
    mesh = anatomical_surface(mask, np.eye(4))
    assert not mesh.is_watertight
    assert mesh.vertices[:,2].min() == 0


def test_wrong_patient_and_shifted_grid_are_rejected():
    with pytest.raises(ValueError):
        require_same_patient(dict(case_id='A',source_ct_sha256='1'),dict(case_id='B',source_ct_sha256='1'))
    with pytest.raises(ValueError):
        require_same_patient(dict(case_id='A',source_ct_sha256='1'),dict(case_id='A',source_ct_sha256='2'))
    affine = np.eye(4); affine[0,3] = 1
    with pytest.raises(ValueError):
        require_same_grid(nib.Nifti1Image(np.zeros((3,3,3)),np.eye(4)),nib.Nifti1Image(np.zeros((3,3,3)),affine))


def test_ply_roundtrip_retains_geometry_and_colors(tmp_path):
    import trimesh
    mask = np.zeros((8,8,8), dtype=bool); mask[2:6,2:6,2:6] = True
    mesh = anatomical_surface(mask, np.diag([.5,1,2,1]))
    mesh.visual.vertex_colors = confidence_colors(np.linspace(0,1,len(mesh.vertices)))
    mesh.export(tmp_path/'test.ply')
    loaded = trimesh.load_mesh(tmp_path/'test.ply', process=False)
    assert geometry_digest(mesh) == geometry_digest(loaded)
    assert np.array_equal(mesh.visual.vertex_colors, loaded.visual.vertex_colors)


def test_existing_confidence_pipeline_uses_full_surface_and_rejects_grid_mismatch(tmp_path):
    import trimesh
    from src.geometry.confidence import run_surface_confidence
    mask = np.zeros((8,8,8), dtype=bool); mask[2:6,2:6,2:6] = True
    mesh = anatomical_surface(mask, np.eye(4))
    p4=tmp_path/'phase4';p7=tmp_path/'phase7';output=tmp_path/'result'
    entropy_path=p4/'cases'/'unit_fixture'/'predictive_entropy.nii.gz'
    entropy_path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros(mask.shape,dtype=np.float32),np.eye(4)),entropy_path)
    for organ in ('brain','skull'):
        target=p7/'cases'/'unit_fixture'/organ;target.mkdir(parents=True)
        mesh.export(target/'baseline_full_resolution.ply')
        # No adaptive file: the full anatomical surface must be sufficient.
    records=[dict(case_id='unit_fixture',shape=list(mask.shape),ct_affine=np.eye(4).tolist())]
    cfg=dict(interpolation_order=1,colormap='RdYlGn',confidence_definition='one_minus_normalized_predictive_entropy')
    run_surface_confidence(records,p4,p7,cfg,output,representative_case='unit_fixture',progress=lambda _:None)
    colored=trimesh.load_mesh(output/'cases/unit_fixture/skull/confidence_colored.ply',process=False)
    assert geometry_digest(colored)==geometry_digest(mesh)
    assert (colored.visual.vertex_colors==[0,255,0,255]).all()
    shifted=np.eye(4);shifted[0,3]=5
    nib.save(nib.Nifti1Image(np.zeros(mask.shape,dtype=np.float32),shifted),entropy_path)
    with pytest.raises(ValueError,match='Entropy grid'):
        run_surface_confidence(records,p4,p7,cfg,output,representative_case='unit_fixture')


def test_missing_confidence_does_not_create_nan_envelope_vertices(tmp_path):
    import trimesh
    from src.geometry.confidence import run_confidence_envelopes
    mask=np.zeros((8,8,8),dtype=bool);mask[2:6,2:6,2:6]=True
    mesh=anatomical_surface(mask,np.eye(4))
    for organ in ('brain','skull'):
        folder=tmp_path/'phase8/cases/unit_fixture'/organ;folder.mkdir(parents=True)
        mesh.export(folder/'confidence_colored.ply')
        np.savez(folder/'vertex_confidence.npz',uncertainty=np.full(len(mesh.vertices),np.nan))
    report=run_confidence_envelopes([dict(case_id='unit_fixture')],tmp_path/'phase8',
        dict(maximum_offset_mm=3,interpretation='not_statistical_confidence_interval'),tmp_path/'phase9',
        representative_case='unit_fixture',progress=lambda _:None)
    envelope=trimesh.load_mesh(tmp_path/'phase9/cases/unit_fixture/skull/outer_envelope.ply',process=False)
    assert geometry_digest(envelope)==geometry_digest(mesh)
    assert report['cases'][0]['organs']['skull']['unavailable_offset_vertex_fraction']==1
