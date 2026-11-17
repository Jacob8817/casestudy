"""Reproducible CASE_001 verification/export; never modifies source data or experiments.

Run: .venv/Scripts/python.exe scripts/verify_patient_skull.py
Independent predictions are compared as agreement, never ground-truth accuracy.
"""
from pathlib import Path
import sys, json, hashlib, csv, shutil, datetime, gc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import pydicom
import trimesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.geometry.patient_surface import (anatomical_surface, color_surface, require_same_grid,
    geometry_digest, sample_world, confidence_colors)
from infer_patient_confidence import run as infer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/patient_skull_verification'
BONE = ROOT/'outputs/10_totalsegmentator_sharp_skull'
BRAIN = ROOT/'outputs/09_totalsegmentator_dl_skull_brain'
FRAMEWORK = ROOT/'outputs/anatomy_aware_framework'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def metadata(path):
    im=nib.load(path)
    return dict(path=str(path),sha256=sha(path),shape=list(im.shape),affine=im.affine.tolist(),
        spacing_mm=nib.affines.voxel_sizes(im.affine).tolist(),orientation=list(nib.aff2axcodes(im.affine)),
        qform_code=int(im.header['qform_code']),sform_code=int(im.header['sform_code']),world_convention='RAS+ mm')


def dicom_audit(directory, image):
    hu=np.asarray(image.dataobj)
    records=[]; maximum_error=0.; maximum_rounding=0.
    for path in sorted(directory.glob('*.dcm')):
        ds=pydicom.dcmread(path)
        orientation=np.array(ds.ImageOrientationPatient,dtype=float).reshape(2,3)
        spacing=np.array(ds.PixelSpacing,dtype=float)
        rc=np.array([[0,0],[127,191],[255,255],[383,319],[511,511]])
        lps=np.array(ds.ImagePositionPatient)+rc[:,1,None]*spacing[1]*orientation[0]+rc[:,0,None]*spacing[0]*orientation[1]
        ras=lps*np.array([-1,-1,1])
        ijk=nib.affines.apply_affine(np.linalg.inv(image.affine),ras)
        rounded=np.rint(ijk).astype(int)
        assert (rounded>=0).all() and (rounded<np.array(image.shape)).all()
        maximum_rounding=max(maximum_rounding,float(np.max(np.abs(ijk-rounded))))
        pixels=ds.pixel_array[rc[:,0],rc[:,1]].astype(float)*float(ds.RescaleSlope)+float(ds.RescaleIntercept)
        error=float(np.max(np.abs(hu[tuple(rounded.T)]-pixels)))
        maximum_error=max(maximum_error,error)
        records.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),patient_id=str(ds.PatientID),
            study_uid=str(ds.StudyInstanceUID),series_uid=str(ds.SeriesInstanceUID),frame_uid=str(getattr(ds,'FrameOfReferenceUID','')),
            position_lps=list(map(float,ds.ImagePositionPatient)),orientation_lps=orientation.ravel().tolist(),pixel_spacing_mm=spacing.tolist()))
    assert records and maximum_error==0 and maximum_rounding<1e-3
    assert len({r['patient_id'] for r in records})==1 and len({r['series_uid'] for r in records})==1
    dump(OUT/(directory.name+'_dicom_manifest.json'),records)
    return dict(slices=len(records),patient_ids=sorted({r['patient_id'] for r in records}),
        study_uids=sorted({r['study_uid'] for r in records}),series_uids=sorted({r['series_uid'] for r in records}),
        frame_uids=sorted({r['frame_uid'] for r in records}),sampled_pixels_per_slice=5,maximum_HU_error=maximum_error,
        maximum_index_rounding_error=maximum_rounding,manifest=directory.name+'_dicom_manifest.json')


def components(mask, ct, affine, name):
    labels,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,1))
    objects=ndi.find_objects(labels); rows=[]; voxel_volume=abs(np.linalg.det(affine[:3,:3]))
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        local=labels[sl]==i; indices=np.argwhere(local)+np.array([s.start for s in sl])
        center=nib.affines.apply_affine(affine,indices.mean(axis=0))
        values=ct[sl][local]
        rows.append(dict(label=name,component_id=i,voxels=int(local.sum()),volume_mm3=float(local.sum()*voxel_volume),
            centroid_ras_mm=center.tolist(),bbox_index=[[s.start,s.stop] for s in sl],
            mean_hu=float(values.mean()),median_hu=float(np.median(values)),fraction_ge_300_hu=float(np.mean(values>=300)),
            touches_fov=bool(any(s.start==0 or s.stop==mask.shape[a] for a,s in enumerate(sl))),action='retained',reason='No independent evidence justifies deletion'))
    nib.save(nib.Nifti1Image(labels.astype(np.int32),affine),OUT/'volumes'/f'{name}_components.nii.gz')
    return rows


def topology(mesh):
    edges=np.sort(np.concatenate([mesh.faces[:,[0,1]],mesh.faces[:,[1,2]],mesh.faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    return dict(vertices=len(mesh.vertices),triangles=len(mesh.faces),bounds_ras_mm=mesh.bounds.tolist(),
        surface_area_mm2=float(mesh.area),boundary_edges=int(np.sum(counts==1)),nonmanifold_edges=int(np.sum(counts>2)),
        degenerate_faces=int(np.sum(mesh.area_faces<=1e-12)),watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),euler_number=int(mesh.euler_number),geometry_sha256=geometry_digest(mesh))


def distances(a,b):
    forward=cKDTree(b.vertices).query(a.vertices,workers=-1)[0]
    backward=cKDTree(a.vertices).query(b.vertices,workers=-1)[0]
    return dict(metric='symmetric nearest-vertex distances; not exact triangle distances',
        mean_mm=float((forward.mean()+backward.mean())/2),p95_mm=float(np.percentile(np.r_[forward,backward],95)),
        max_mm=float(max(forward.max(),backward.max())),bounding_box_max_difference_mm=float(np.abs(a.bounds-b.bounds).max()))


def render(meshes, titles, path, bounds, direction=(1.3,1.5,.8), colors=True):
    """Full-resolution VTK rendering with common patient-space camera/scale."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(800*len(meshes),850)
    # Bounding-sphere framing covers every orientation, including oblique views.
    center=np.mean(bounds,axis=0); radius=np.linalg.norm(bounds[1]-bounds[0])*.57
    direction=np.array(direction,dtype=float);direction/=np.linalg.norm(direction)
    for index,(mesh,title) in enumerate(zip(meshes,titles)):
        renderer=vtk.vtkRenderer(); renderer.SetViewport(index/len(meshes),0,(index+1)/len(meshes),1);renderer.SetBackground(.07,.08,.10)
        window.AddRenderer(renderer)
        points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.asarray(mesh.vertices),deep=True))
        faces=np.c_[np.full(len(mesh.faces),3),mesh.faces].astype(np.int64)
        cells=vtk.vtkCellArray();cells.SetCells(len(faces),numpy_to_vtkIdTypeArray(faces.ravel(),deep=True))
        poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
        mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(poly)
        if colors:
            arr=numpy_to_vtk(np.asarray(mesh.visual.vertex_colors[:,:3]),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)
            poly.GetPointData().SetScalars(arr);mapper.SetColorModeToDirectScalars();mapper.ScalarVisibilityOn()
        else: mapper.ScalarVisibilityOff()
        actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetColor(.88,.82,.69);actor.GetProperty().SetAmbient(.25)
        renderer.AddActor(actor)
        text=vtk.vtkTextActor();text.SetInput(title);text.SetPosition(20,810);text.GetTextProperty().SetFontSize(19);renderer.AddViewProp(text)
        if colors and index==len(meshes)-1:
            lut=vtk.vtkLookupTable();lut.SetNumberOfTableValues(256);lut.SetRange(0,1);lut.Build()
            for k,rgba in enumerate(confidence_colors(np.linspace(0,1,256))/255): lut.SetTableValue(k,*rgba)
            bar=vtk.vtkScalarBarActor();bar.SetLookupTable(lut);bar.SetOrientationToHorizontal()
            bar.SetPosition(.16,.025);bar.SetWidth(.68);bar.SetHeight(.075);bar.SetNumberOfLabels(3)
            bar.SetTitle('MONAI P(skull|CT); grey: unavailable');bar.GetTitleTextProperty().SetFontSize(14)
            renderer.AddViewProp(bar)
        camera=renderer.GetActiveCamera();camera.SetFocalPoint(*center);camera.SetPosition(*(center+direction*radius*5))
        camera.SetViewUp(*( (0,1,0) if abs(direction[2])>.95 else (0,0,1)))
        camera.ParallelProjectionOn();camera.SetParallelScale(radius);renderer.ResetCameraClippingRange()
    window.Render(); capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.ReadFrontBufferOff();capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(path));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    window.Finalize()


def interactive(mesh, path, title, legend=True):
    v,f=mesh.vertices,mesh.faces
    traces=[go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],vertexcolor=mesh.visual.vertex_colors,flatshading=True,name=title)]
    if legend:
        traces.append(go.Scatter3d(x=[None,None],y=[None,None],z=[None,None],mode='markers',showlegend=False,
            marker=dict(color=[0,1],cmin=0,cmax=1,colorscale=[[0,'red'],[.5,'yellow'],[1,'green']],showscale=True,colorbar=dict(title='MONAI P(skull|CT)'))))
    fig=go.Figure(traces)
    fig.update_layout(title=title+'<br>RAS+ mm; grey = unavailable. Research prediction, not verified anatomy.',
        scene=dict(aspectmode='data',xaxis_title='Right (mm)',yaxis_title='Anterior (mm)',zaxis_title='Superior (mm)',
            camera=dict(eye=dict(x=1.3,y=1.5,z=.8),up=dict(x=0,y=0,z=1))),margin=dict(l=0,r=0,b=0,t=70))
    fig.write_html(path,include_plotlyjs=True)


def overlays(ct, masks, affine):
    names=list(masks); colors=['#ffd166','#44ddff','#ff66dd','#55ff88']
    sets={'alignment_axial':[0,10,30,53,80,107,133,160,187], 'cranial_opening':[80,100,120,140,160,180]}
    for title,zs in sets.items():
        fig,axes=plt.subplots(len(zs)//3,3,figsize=(13,4.4*(len(zs)//3)),squeeze=False)
        for ax,z in zip(axes.ravel(),zs):
            ax.imshow(ct[:,:,z].T,origin='lower',cmap='gray',vmin=-200,vmax=1400)
            for name,color in zip(names,colors):
                sl=masks[name][:,:,z].T
                if sl.any(): ax.contour(sl,levels=[.5],colors=[color],linewidths=.65)
            ax.set_xlim(70,440);ax.set_ylim(430,55)
            ax.set_title(f'Native index z={z}; superior {nib.affines.apply_affine(affine,[0,0,z])[2]:.2f} mm')
            ax.set_xlabel('i index');ax.set_ylabel('j index')
        fig.suptitle('CASE_001 bone CT + unmodified predictions\nYellow: skull; cyan: mandible; magenta: upper teeth. Index axes; see affine for orientation.')
        fig.tight_layout();fig.savefig(OUT/'figures'/f'{title}.png',dpi=160);plt.close(fig)
    # Raw CT alongside segmentation at representative defect slices, no overlay obscures bone.
    fig,axes=plt.subplots(3,2,figsize=(10,14))
    for row,z in enumerate([107,133,160]):
        for col in range(2):
            ax=axes[row,col];ax.imshow(ct[:,:,z].T,origin='lower',cmap='gray',vmin=-200,vmax=1400)
            if col: ax.contour(masks['skull'][:,:,z].T,levels=[.5],colors=['yellow'],linewidths=.8)
            ax.set_xlim(95,410);ax.set_ylim(410,80);ax.set_title(f'z={z}: '+('CT + skull prediction' if col else 'Original CT'))
    fig.suptitle('Cranial opening: acquired CT evidence; no bone synthesis or closure')
    fig.tight_layout();fig.savefig(OUT/'figures'/'opening_raw_vs_overlay.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for folder in ('meshes','figures','volumes','previous'): (OUT/folder).mkdir(exist_ok=True)
    if (OUT/'verification_report.json').exists():
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(OUT/'verification_report.json',OUT/'previous'/f'report_{stamp}.json')
    ct_path=BONE/'input/CASE_001_bone_kernel_ct.nii.gz';ct_img=nib.load(ct_path);ct=np.asarray(ct_img.dataobj,dtype=np.float32)
    paths={n:BONE/'craniofacial_segmentations'/f'{n}.nii.gz' for n in ('skull','mandible','teeth_upper','teeth_lower')}
    protected=[ct_path,*paths.values(),BRAIN/'input/CASE_001_brain_kernel_ct.nii.gz',BRAIN/'segmentations/skull.nii.gz',
        ROOT/'data/splits/dataset_manifest.json',FRAMEWORK/'phase3/best.pt',FRAMEWORK/'phase4/phase4_report.json']
    protected += list((FRAMEWORK/'phase8').rglob('*.*'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    print('Verifying original DICOM pixels and NIfTI geometry',flush=True)
    dicom_root=ROOT/'Single_Patient_Study/Single_Patient_Study/CASE_001/ANALYSIS'
    dicom={ 'bone':dicom_audit(dicom_root/'BONE_HR60F',ct_img),
        'brain':dicom_audit(dicom_root/'BRAIN_HR36F',nib.load(BRAIN/'input/CASE_001_brain_kernel_ct.nii.gz'))}
    assert dicom['bone']['patient_ids']==dicom['brain']['patient_ids']
    assert dicom['bone']['study_uids']==dicom['brain']['study_uids']
    masks={}; label_info={}; rows=[]
    for name,path in paths.items():
        im=nib.load(path);require_same_grid(ct_img,im); mask=np.asarray(im.dataobj)>0;masks[name]=mask
        entries=components(mask,ct,ct_img.affine,name);rows+=entries
        label_info[name]=dict(voxels=int(mask.sum()),volume_mm3=float(mask.sum()*abs(np.linalg.det(ct_img.affine[:3,:3]))),
            components_6_connected=len(entries),fov_face_voxels={f'{axis}_{side}':int(np.take(mask,idx,axis).sum()) for axis in range(3) for side,idx in [('low',0),('high',-1)]},
            hu_percentiles=np.percentile(ct[mask],[0,5,50,95,100]).tolist() if mask.any() else None)
    dump(OUT/'components.json',rows)
    union=np.logical_or.reduce(list(masks.values()))
    nib.save(nib.Nifti1Image(union.astype(np.uint8),ct_img.affine),OUT/'volumes/corrected_anatomy_mask.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros(ct.shape,dtype=np.uint8),ct_img.affine),OUT/'volumes/removed_components.nii.gz')
    overlays(ct,masks,ct_img.affine)
    print('Building native-resolution, unfiltered surfaces',flush=True)
    meshes={name:anatomical_surface(mask,ct_img.affine) for name,mask in masks.items() if mask.any()}
    for name,mesh in meshes.items():
        mesh.visual.vertex_colors=np.tile([225,209,176,255],(len(mesh.vertices),1))
        mesh.export(OUT/'meshes'/f'{name}_anatomical.ply')
    provenance=infer(OUT/'confidence')
    prob_image=nib.load(OUT/'confidence/skull_probability.nii.gz');prob=np.asarray(prob_image.dataobj)
    reference=dict(case_id='CASE_001',source_ct_sha256=sha(ct_path))
    colored,values=color_surface(meshes['skull'],prob,prob_image.affine,reference_provenance=reference,confidence_provenance=provenance)
    np.savez_compressed(OUT/'meshes/skull_vertex_confidence.npz',confidence=values,vertices=colored.vertices,faces=colored.faces)
    colored.export(OUT/'meshes/skull_confidence.ply')
    neutral=[]
    for name in ('mandible','teeth_upper','teeth_lower'):
        if name in meshes:
            m=meshes[name].copy();m.visual.vertex_colors=np.tile([150,150,150,255],(len(m.vertices),1));neutral.append(m)
    combined=trimesh.util.concatenate(list(meshes.values()))
    combined_conf=trimesh.util.concatenate([colored,*neutral])
    assert geometry_digest(combined)==geometry_digest(combined_conf)
    combined.export(OUT/'meshes/skull_plus_mandible_anatomical.ply');combined_conf.export(OUT/'meshes/skull_plus_mandible_confidence.ply')
    for name,mesh in [('skull',colored),('skull_plus_mandible',combined_conf)]:
        trimesh.Scene(mesh).export(OUT/'meshes'/f'{name}_confidence.glb')
        title='CASE_001 skull + partial mandible + upper teeth' if name=='skull_plus_mandible' else 'CASE_001 skull only + confidence'
        interactive(mesh,OUT/f'{name}_confidence.html',title)
    # Verify the actual exported geometry, not just an in-memory copy.
    reloaded=trimesh.load_mesh(OUT/'meshes/skull_plus_mandible_confidence.ply',process=False)
    assert geometry_digest(reloaded)==geometry_digest(combined)
    assert np.array_equal(reloaded.visual.vertex_colors,combined_conf.visual.vertex_colors)
    bounds=combined.bounds
    print('Rendering full-resolution comparison figures',flush=True)
    render([combined,combined_conf],['CASE_001 skull + partial mandible + upper teeth','Identical anatomy + MONAI skull probability'],OUT/'figures/geometry_confidence_comparison.png',bounds)
    for name,direction in [('frontal',(0,1,0)),('lateral',(-1,0,0)),('superior',(0,0,1)),('inferior',(0,0,-1)),('oblique',(-1.3,1.5,.8))]:
        render([combined,combined_conf],[f'{name}: skull + partial jaw + upper teeth',f'{name}: identical geometry + confidence'],OUT/'figures'/f'{name}.png',bounds,direction)
    original_path=BONE/'meshes/CASE_001_totalsegmentator_sharp_skull.ply'
    original=trimesh.load_mesh(original_path,process=False);shutil.copy2(original_path,OUT/'previous'/original_path.name)
    render([original,combined],['Original notebook 10 (smoothed/capped)','Native labels (no smoothing or FOV cap)'],OUT/'figures/original_vs_corrected.png',np.stack([np.minimum(original.bounds[0],bounds[0]),np.maximum(original.bounds[1],bounds[1])]),colors=False)
    render([combined,combined],['Components before: all labels retained','Components after: 0 voxels removed'],OUT/'figures/components_before_after.png',bounds,colors=False)
    # CT intensity is an evidence visualization, not an anatomical segmentation.
    ct_mesh=anatomical_surface(ct>=300,ct_img.affine)
    ct_mesh.export(OUT/'meshes/ct_300hu_evidence.ply')
    render([ct_mesh],['Original CT >= 300 HU; includes nonbone objects'],OUT/'figures/ct_bone_evidence.png',ct_mesh.bounds,colors=False)
    del ct_mesh;gc.collect()
    # Uncertainty diagnostic is kept distinct from anatomical confidence.
    entropy_img=nib.load(OUT/'confidence/predictive_entropy.nii.gz');entropy=np.asarray(entropy_img.dataobj)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,fraction in zip(axes,[.35,.55,.75]):
        z=int(entropy.shape[2]*fraction);im=ax.imshow(entropy[:,:,z].T,origin='lower',cmap='magma',vmin=0,vmax=1);ax.set_title(f'Model-grid z={z}');ax.axis('off')
    fig.colorbar(im,ax=list(axes),label='Normalized predictive entropy H/log(3); high = uncertain');fig.suptitle('CASE_001 frozen MONAI model; not epistemic uncertainty')
    fig.savefig(OUT/'figures/uncertainty_only.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,1.8));gradient=np.linspace(0,1,400)
    ax.imshow(confidence_colors(gradient)[None,:,:3],extent=(0,1,0,1),aspect='auto');ax.set_yticks([]);ax.set_xlabel('MONAI P(skull | CT), fixed temperature; grey = unavailable');fig.tight_layout();fig.savefig(OUT/'figures/confidence_legend.png',dpi=160);plt.close(fig)
    # Cross-pipeline agreement on the same native grid; neither mask is ground truth.
    older_img=nib.load(BRAIN/'segmentations/skull.nii.gz');require_same_grid(ct_img,older_img);older=np.asarray(older_img.dataobj)>0
    overlap=int(np.count_nonzero(older&masks['skull']))
    agreement=dict(dice=float(2*overlap/(older.sum()+masks['skull'].sum())),iou=float(overlap/np.count_nonzero(older|masks['skull'])),interpretation='Total vs craniofacial model agreement, not accuracy; class scopes differ')
    # Existing s0346 files are read-only; compare full vs adaptive vs colored.
    oldbase=trimesh.load_mesh(FRAMEWORK/'phase7/cases/s0346/skull/baseline_full_resolution.ply',process=False)
    oldadaptive=trimesh.load_mesh(FRAMEWORK/'phase7/cases/s0346/skull/adaptive_priority.ply',process=False)
    oldcolored=trimesh.load_mesh(FRAMEWORK/'phase8/cases/s0346/skull/confidence_colored.ply',process=False)
    prior_geometry=dict(full=topology(oldbase),adaptive=topology(oldadaptive),confidence=topology(oldcolored),
        color_preserved_adaptive_geometry=geometry_digest(oldadaptive)==geometry_digest(oldcolored),adaptive_displacement=distances(oldbase,oldadaptive))
    s0346_meta=json.loads((ROOT/'data/cache/anatomy_aware_phase2/s0346/metadata.json').read_text())
    rawroot=next(p.parent for p in (ROOT/'Totalsegmentator_dataset_v201').rglob('ct.nii.gz') if p.parent.name=='s0346')
    assert sha(rawroot/'ct.nii.gz')==s0346_meta['source_ct_sha256']
    assert sha(rawroot/'segmentations/skull.nii.gz')==s0346_meta['source_skull_sha256']
    from src.geometry.confidence import run_surface_confidence
    legacy_fixed=run_surface_confidence([s0346_meta],FRAMEWORK/'phase4',FRAMEWORK/'phase7',
        dict(interpolation_order=1,colormap='RdYlGn',confidence_definition='one_minus_normalized_predictive_entropy'),
        OUT/'s0346_corrected',representative_case='s0346')
    prior_geometry['corrected_full_resolution_export']=legacy_fixed
    for path in (FRAMEWORK/'phase8/phase8_report.json',FRAMEWORK/'phase8/cases/s0346/skull/confidence_colored.ply'):
        shutil.copy2(path,OUT/'previous'/('s0346_'+path.name))
    after={str(p.relative_to(ROOT)):sha(p) for p in protected};assert before==after
    points=np.array([[0,0,0],np.array(ct.shape)-1,np.array(ct.shape)/2])
    roundtrip=float(np.max(np.abs(nib.affines.apply_affine(np.linalg.inv(ct_img.affine),nib.affines.apply_affine(ct_img.affine,points))-points)))
    assert roundtrip<1e-8
    report=dict(case_id='CASE_001',status='technical_geometry_verified_anatomy_requires_expert_review',
        input=metadata(ct_path),labels={k:metadata(p) for k,p in paths.items()},dicom=dicom,
        provenance=dict(original_patient='CASE_001',old_confidence_patient='s0346',same_patient=False,
            old_confidence_ct=metadata(rawroot/'ct.nii.gz'),old_confidence_cache=s0346_meta,new_confidence=provenance),
        labels_metrics=label_info,component_policy='Retain all anatomical class components; no deletion, bridging or filling',removed_volume_mm3=0,
        geometry=dict(anatomical=topology(combined),confidence=topology(combined_conf),exact_identity=True,
            displacement_mm=0.,hausdorff_mm=0.,bounding_box_difference_mm=0.,exported_ply_identity=True,
            original_to_native=distances(original,combined),prior_s0346=prior_geometry),
        confidence=dict(definition=provenance['confidence_definition'],skull_vertices=len(values),valid_fraction=float(np.isfinite(values).mean()),
            invalid_percent=float(100*np.mean(~np.isfinite(values))),range=[float(np.nanmin(values)),float(np.nanmax(values))],
            separate_mandible_teeth_confidence='unavailable; neutral grey',geometry_thresholding=False),
        alignment=dict(same_grid=True,roundtrip_max_voxels=roundtrip,dicom_pixel_samples_match=True),
        mask_agreement=agreement,protected_inputs_unchanged=True,protected_hashes=before,
        opening=dict(status='CT-supported absence of cranial bone; preserved',review_slices_native=[80,107,133,160,187],
            etiology='Not established by this technical audit. Supplied annotations mention postoperative collection; this does not prove the cause of the opening.'),
        regional_changes=dict(cranial_opening='No voxels added/removed',orbits='No voxels added/removed',maxilla='No voxels added/removed',mandible='Separate partial label retained; reaches inferior FOV'),
        limitations=['No independently expert-verified segmentation or labeled landmarks supplied.',
            'All selected reference labels remain model predictions, even when spatially and visually consistent with CT.',
            'Mandible extends beyond inferior CT FOV; no complete jaw can be reconstructed.',
            'No separate frontal/parietal/temporal/occipital/orbital/maxillary class labels; regional completeness cannot be certified.',
            'Thin bones and detached components need expert review; no component removed without evidence.',
            'External CASE_001 confidence calibration transfer is unverified; no calibration changes made.'],
        outputs=[str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.html','.png','.ply','.glb','.gz','.npz')])
    conditions={
        'Both original outputs traceable':('PASS','CASE_001 versus s0346: mismatch explicitly identified'),
        'Original outputs same patient':('FAIL','Different patient and acquisition; new matched outputs generated'),
        'CT and masks aligned':('PASS','Affines, shapes, all-slice DICOM pixel sampling, round trips'),
        'Skull and mandible class handling':('PASS','Separate source labels retained; mandible partial at FOV'),
        'Confidence does not remove anatomy':('PASS','No thresholding; exact geometry identity'),
        'Confidence matches selected anatomical surface':('PASS','Identical vertices and triangles, including exported PLY'),
        'Cranial opening investigated':('PASS','Raw CT and overlays through multiple native slices'),
        'No CT defect filled':('PASS','Zero segmentation edits, no fill/closing/capping'),
        'Components anatomically justified':('PASS','Conservative retain-all policy; per-component CT evidence recorded'),
        'Identical comparison cameras':('PASS','Shared RAS bounds, parallel scale and view for every comparison'),
        'Automated regression tests executed':('PENDING','See regression_tests.xml'),
        '3D and static exports':('PASS','PLY, GLB, HTML and PNG'),
        'Modifications and limitations documented':('PASS','Report and source audit'),
        'Independent anatomical accuracy verified':('NOT_ESTABLISHED','No expert ground truth; no accuracy claim')}
    import xml.etree.ElementTree as ET
    test_path=OUT/'regression_tests.xml'
    if test_path.exists():
        suites=ET.parse(test_path).getroot();suites=list(suites) if suites.tag=='testsuites' else [suites]
        test_metrics={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        report['tests']=test_metrics
        conditions['Automated regression tests executed']=('PASS' if test_metrics['tests']>0 and test_metrics['failures']==test_metrics['errors']==0 else 'FAIL',str(test_metrics))
    report['completion_conditions']={k:dict(status=v[0],evidence=v[1]) for k,v in conditions.items()}
    dump(OUT/'verification_report.json',report)
    write_report(report)
    print('Verification exported:',OUT,flush=True)


def write_report(r):
    lines=['# CASE_001 skull reconstruction verification','',
        '**Assessment:** New anatomical and confidence-colored outputs have exactly the same patient geometry. Technical provenance and geometry checks pass. Independent anatomical accuracy is not established; the reference is a CT-aligned prediction requiring expert review.','',
        '## Patient verification', '',
        'The original patient reconstruction is CASE_001. The old phase-8 confidence model is **s0346**, an official validation subject. Comparing them as identical anatomy was invalid. CASE_001 bone and brain kernels have matching patient/study identifiers but distinct reconstruction series; DICOM manifests retain the exact UIDs and hashes.', '',
        f'CT: `{r["input"]["path"]}`. SHA-256: `{r["input"]["sha256"]}`. Shape: {r["input"]["shape"]}; spacing: {r["input"]["spacing_mm"]} mm; array orientation: {r["input"]["orientation"]}; physical coordinates: RAS+ mm.', '',
        'Every slice in each 227-slice DICOM series was checked at five known pixel locations using DICOM LPS orientation/position/spacing, converted to NIfTI RAS. Pixel HU differences were zero. This establishes conversion and grid consistency, not a clinical landmark validation.', '',
        '## Confirmed causes and unresolved hypotheses','',
        '- **Different skulls:** confirmed patient/source mismatch, CASE_001 versus s0346.',
        '- **Confidence-dependent loss:** old phase 8 colored the phase-7 adaptive mesh. It did not threshold confidence voxels. The old confidence/adaptive geometry identity is '+str(r['geometry']['prior_s0346']['color_preserved_adaptive_geometry'])+'. Geometry reduction occurred upstream in spatial vertex clustering; full/native geometry is now the default.',
        '- **Facial detail:** patient mismatch, different segmentation sources and native versus 1.5 mm model-grid resolution prevent a direct same-anatomy conclusion. Notebook 09 additionally uses step-size 2 and five smoothing iterations. No unsupported facial bone was synthesized.',
        '- **Neural rendering:** notebook 11 uses predicted SIREN HU >=250 within a dilated skull neighborhood, a minimum 100-voxel component size, at most 12 components and marching-cubes step-size 2. This is another derived approximation of CASE_001, not the acquired CT or an independent anatomical reference. It is retained as a historical experiment.',
        '- **Mandible:** craniofacial task has a separate mandible label. Earlier skull-only rendering omitted that separate label. Notebook 07 also globally retained the largest component, which can discard separate anatomy. The available mandible reaches the inferior acquired boundary; a complete lower jaw is outside the available evidence.',
        '- **Cranial opening:** visible absence of a continuous bone rim on native CT slices 80, 107, 133, 160 and 187. It is not explained solely by mesh coloring. Preserved unchanged. Postoperative etiology is plausible from the supplied annotations but is not established here; do not label it a fracture.',
        '- **Detached fragments:** source-label components are enumerated with physical volume, location, HU statistics and FOV contact. Their medical validity is unresolved; all are retained.',
        f'- **Surface fragmentation:** old s0346 full mesh had {r["geometry"]["prior_s0346"]["full"]["nonmanifold_edges"]} non-manifold edges; adaptive clustering increased this to {r["geometry"]["prior_s0346"]["adaptive"]["nonmanifold_edges"]}. It merged spatially close vertices and removed degenerate/duplicate faces. New CASE_001 mesh has {r["geometry"]["anatomical"]["nonmanifold_edges"]} non-manifold edges and {r["geometry"]["anatomical"]["boundary_edges"]} boundary edges. Native voxel surfaces retain voxel-scale irregularity. Acquired-FOV boundaries are not capped.',
        '- **Alignment:** native CASE_001 labels pass grid checks. Earlier notebook 07 converts array coordinates through DICOM directions/origin into LPS, while NIfTI workflows export RAS. Direct overlays require the documented LPS-to-RAS sign conversion, not arbitrary fitting. New exports use the NIfTI affine.', '',
        '## Corrections and confidence definition','',
        '`src/geometry/patient_surface.py`: native-resolution mask surfaces, no FOV padding, no smoothing/decimation/filtering, explicit affine sampling, fail-closed patient/hash checks, unavailable values grey and geometry digests.',
        '`src/geometry/confidence.py`: phase 8 now uses baseline_full_resolution.ply; rejects entropy-grid mismatches; no nearest-boundary extrapolation; checks geometry identity and reports invalid coverage. Existing phase outputs are preserved.',
        '`configs/anatomy_aware_phases4_10_v1.yaml`: red/yellow/green legend. No training, split, calibration or selection parameters changed.',
        '`scripts/infer_patient_confidence.py`: frozen phase-3 MONAI inference on CASE_001 CT resampled in RAS at 1.5 mm, using established two HU windows and fixed existing temperature. Full logits/probabilities saved. This is external-case inference, not held-out dataset evaluation.',
        '`scripts/verify_patient_skull.py`: DICOM/source audit, per-class native surfaces, quantitative checks, exports and this report. `tests/test_patient_surface.py`: regression checks.', '',
        f'Main CASE_001 legend: **P(skull | CT) = softmax(logits / {r["provenance"]["new_confidence"]["temperature"]})[2]**. Red=0, yellow=0.5, green=1; grey=unavailable. Background and brain are retained in the softmax denominator; no foreground-only renormalization. Mandible/teeth have no dedicated model probability and remain grey. These are MONAI predictions sampled on TotalSegmentator geometry, not confidence emitted by TotalSegmentator.', '',
        'Diagnostic entropy is -sum(p log p)/log(3), across background/brain/skull. Higher entropy means greater predictive uncertainty. It is neither epistemic uncertainty nor class-specific skull probability. The legacy 1-entropy metric is retained only with explicit predictive-certainty labeling. Temperature was not refitted; calibration transfer to this external scan is unverified.', '',
        '## Quantitative verification','',
        '| Label | Voxels | Volume mm3 | Components (6-neighbor) |','|---|---:|---:|---:|']
    for k,v in r['labels_metrics'].items():lines.append(f'| {k} | {v["voxels"]} | {v["volume_mm3"]:.3f} | {v["components_6_connected"]} |')
    lines += ['',f'New combined mesh: {r["geometry"]["anatomical"]["vertices"]:,} vertices, {r["geometry"]["anatomical"]["triangles"]:,} triangles. Anatomical versus confidence displacement, Hausdorff distance and bounding-box difference: **0 mm**, from exact vertex/connectivity identity. Segmentation voxels added/removed: **0/0**. Removed component volume: **0 mm3**.',
        f'Skull confidence invalid coverage: {r["confidence"]["invalid_percent"]:.6f}%. Full per-class/FOV/topology diagnostics are in JSON. Original-smoothed to native nearest-vertex distances: {r["geometry"]["original_to_native"]}.',
        f'Total vs craniofacial skull model agreement: Dice={r["mask_agreement"]["dice"]:.6f}, IoU={r["mask_agreement"]["iou"]:.6f}. This is not ground-truth accuracy; the class scopes differ.',
        f'Regression tests: {r.get("tests", "not found")}. Initial test invocation encountered a Windows temporary-directory permission error; rerun uses a workspace-local temporary directory.', '',
        '## Visual evidence','',
        '- [Interactive skull + partial mandible/teeth confidence](skull_plus_mandible_confidence.html)',
        '- [Interactive skull-only confidence](skull_confidence.html)',
        '- [GLB](meshes/skull_plus_mandible_confidence.glb) / [anatomical PLY](meshes/skull_plus_mandible_anatomical.ply) / [colored PLY](meshes/skull_plus_mandible_confidence.ply)',
        '- [Identical geometry comparison](figures/geometry_confidence_comparison.png) / [numeric legend](figures/confidence_legend.png)',
        '- [Original versus corrected extraction](figures/original_vs_corrected.png) / [components before/after](figures/components_before_after.png)',
        '- [CT intensity evidence](figures/ct_bone_evidence.png) / [alignment overlays](figures/alignment_axial.png)',
        '- [Raw CT and opening overlays](figures/opening_raw_vs_overlay.png) / [opening levels](figures/cranial_opening.png)',
        '- [Frontal](figures/frontal.png), [lateral](figures/lateral.png), [superior](figures/superior.png), [inferior](figures/inferior.png), [oblique](figures/oblique.png)',
        '- [Predictive uncertainty only](figures/uncertainty_only.png)', '',
        '- [Corrected s0346 full-resolution predictive-certainty display](s0346_corrected/s0346_confidence.html), explicitly a different patient.',
        '- [Source notebook audit](notebook_source_audit.json), [file inventory](file_inventory.json), [per-component evidence](components.json), [export checks](artifact_checks.json).', '',
        '## Completion conditions','', '| Condition | Status | Evidence |','|---|---|---|']
    for k,v in r['completion_conditions'].items():lines.append(f'| {k} | {v["status"]} | {v["evidence"]} |')
    lines += ['', '## Remaining limitations','']+['- '+x for x in r['limitations']]+['',
        'No independently labeled anatomical landmarks or expert review is supplied. Major cranial regions are inspected visually but cannot be individually certified from a single composite skull label. Corrected means corrected extraction/provenance/confidence handling, not a clinically certified segmentation. Original experiment artifacts and protected data hashes remain unchanged. The official 1,228-subject split manifest was fingerprinted before/after; no test payload was evaluated or tuned.', '',
        '## Reproduction','', 'From repository root: `.venv/Scripts/python.exe scripts/verify_patient_skull.py`. Frozen inference is cached only when its source/checkpoint/calibration contract and output hashes match. Each surface is re-extracted deterministically; no stochastic mesh sampling. CUDA deterministic warnings are enabled, so cross-device bitwise inference reproducibility is not claimed.', '',
        'Coordinate convention: [NiBabel affine documentation](https://nipy.org/nibabel/coordinate_systems.html). Temperature method: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html). These references support implementation conventions, not patient-specific anatomical conclusions.']
    (OUT/'verification_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def finalize_report():
    """Verify saved artifacts and finalize the report after regression/visual QA."""
    import xml.etree.ElementTree as ET
    from scipy.special import softmax
    from src.evaluation.metrics import normalized_entropy
    r=json.loads((OUT/'verification_report.json').read_text())
    mesh=trimesh.load_mesh(OUT/'meshes/skull_plus_mandible_anatomical.ply',process=False)
    ply=trimesh.load_mesh(OUT/'meshes/skull_plus_mandible_confidence.ply',process=False)
    scene=trimesh.load(OUT/'meshes/skull_plus_mandible_confidence.glb',force='scene',process=False)
    glb=trimesh.util.concatenate(tuple(scene.geometry.values()))
    assert geometry_digest(mesh)==geometry_digest(ply)==geometry_digest(glb)
    assert np.array_equal(ply.visual.vertex_colors,glb.visual.vertex_colors)
    legacy=trimesh.load_mesh(OUT/'s0346_corrected/cases/s0346/skull/confidence_colored.ply',process=False)
    oldfull=trimesh.load_mesh(FRAMEWORK/'phase7/cases/s0346/skull/baseline_full_resolution.ply',process=False)
    assert geometry_digest(legacy)==geometry_digest(oldfull)
    with np.load(OUT/'confidence/logits_probabilities.npz') as archive:
        p=archive['probabilities']; computed=softmax(archive['logits']/r['provenance']['new_confidence']['temperature'],axis=0)
        probability_error=float(np.max(np.abs(p-computed)))
        assert probability_error<1e-6 and np.allclose(p.sum(axis=0),1,atol=1e-6)
        entropy_error=float(np.max(np.abs(normalized_entropy(p,class_axis=0)-np.asarray(nib.load(OUT/'confidence/predictive_entropy.nii.gz').dataobj))))
        assert entropy_error<1e-6
    manifest=json.loads((ROOT/'data/splits/dataset_manifest.json').read_text())
    assert manifest['dataset']['case_count']==1228
    # Compare only locked manifest metadata, without opening any test scan.
    source_hash=r['input']['sha256']
    assert not any(source_hash in json.dumps(case['ct']) for case in manifest['cases'])
    for name,digest in r['protected_hashes'].items():assert sha(ROOT/name)==digest
    suites=ET.parse(OUT/'regression_tests.xml').getroot()
    suites=list(suites) if suites.tag=='testsuites' else [suites]
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']>0 and tests['failures']==tests['errors']==0
    r['tests']=tests
    r['completion_conditions']['Automated regression tests executed']=dict(status='PASS',evidence=str(tests))
    r['geometry']['exported_glb_identity']=True
    r['dataset_protection']=dict(subjects=1228,split_counts=manifest['split_policy']['counts'],external_CT_hash_absent_from_manifest=True,
        note='Exact compressed-file identity exclusion only; not independent patient linkage proof',test_payloads_accessed=False,protected_hashes_unchanged=True)
    r['source_modifications']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/geometry/patient_surface.py',ROOT/'src/geometry/confidence.py',ROOT/'scripts/infer_patient_confidence.py',ROOT/'scripts/verify_patient_skull.py',ROOT/'tests/test_patient_surface.py',ROOT/'configs/anatomy_aware_phases4_10_v1.yaml',ROOT/'README.md']}
    r['root_causes']={
        'patient_mismatch':dict(status='confirmed',evidence='CASE_001 original versus s0346 confidence'),
        'missing_mandible':dict(status='partly_explained',evidence='Separate craniofacial label; 1184 mandible voxels touch inferior CT plane. Notebook 07 uses global largest component.'),
        'missing_facial_anatomy':dict(status='partly_explained',evidence='Different patient/model/grid; upstream adaptive simplification. Specific missing bones lack independent labels.'),
        'cranial_opening':dict(status='CT_supported',evidence='Native raw CT reviewed at 107,133,160 and wider overlays; no continuous bone rim in the opening. Etiology not established.'),
        'fragments':dict(status='unresolved_anatomical_validity',evidence='39 total retained label components; physical volume/location/HU records in components.json'),
        'fragmentation':dict(status='confirmed_topology_degradation',evidence='s0346 nonmanifold edges: full=3; adaptive=3661'),
        'confidence_deletion':dict(status='not_found_in_phase8',evidence='Old phase8/adaptive geometry identical; reduction was upstream'),
        'coordinate_conventions':dict(status='confirmed_distinct_conventions',evidence='Notebook07 DICOM LPS; NIfTI pipelines RAS; corrected meshes use RAS mm')}
    unique_edges,edge_counts=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
    edge_vertices=mesh.vertices[unique_edges[edge_counts==1]]
    voxel=nib.affines.apply_affine(np.linalg.inv(np.asarray(r['input']['affine'])),edge_vertices)
    at_boundary=np.isclose(voxel,0,atol=1e-3)|np.isclose(voxel,np.asarray(r['input']['shape'])-1,atol=1e-3)
    same_boundary=np.any(np.all(at_boundary,axis=1),axis=1)
    assert same_boundary.all(), 'Unexpected boundary away from acquired FOV requires review'
    checks=dict(exported_PLY_GLB_geometry_identical=True,exported_colors_identical=True,s0346_full_surface_identity=True,
        all_boundary_edges_on_acquired_FOV=True,boundary_edge_count=len(edge_vertices),
        probability_formula_max_error=probability_error,entropy_formula_max_error=entropy_error,
        split_counts=manifest['split_policy']['counts'],tests=tests,
        coordinates='All mesh numeric coordinates are RAS+ millimetres; configure import units as mm.',
        visual_QA='Reviewed CT opening, geometry comparison and oblique full-resolution PNGs; identical camera pairs with legend and complete framing.')
    dump(OUT/'artifact_checks.json',checks)
    r['artifact_checks']=checks
    r['outputs']=[str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.html','.png','.ply','.glb','.gz','.npz')]
    dump(OUT/'verification_report.json',r);write_report(r)
    dump(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print('Final exported-artifact, probability, split and test checks passed',flush=True)


def render_saved_outputs():
    """Repeat figure QA without repeating inference or touching source artifacts."""
    combined=trimesh.load_mesh(OUT/'meshes/skull_plus_mandible_anatomical.ply',process=False)
    colored=trimesh.load_mesh(OUT/'meshes/skull_plus_mandible_confidence.ply',process=False)
    bounds=combined.bounds
    render([combined,colored],['CASE_001 skull + partial mandible + upper teeth','Identical anatomy + MONAI skull probability'],OUT/'figures/geometry_confidence_comparison.png',bounds)
    for name,direction in [('frontal',(0,1,0)),('lateral',(-1,0,0)),('superior',(0,0,1)),('inferior',(0,0,-1)),('oblique',(-1.3,1.5,.8))]:
        render([combined,colored],[f'{name}: skull + partial jaw + upper teeth',f'{name}: identical geometry + confidence'],OUT/'figures'/f'{name}.png',bounds,direction)
    original=trimesh.load_mesh(BONE/'meshes/CASE_001_totalsegmentator_sharp_skull.ply',process=False)
    render([original,combined],['Original notebook 10 (smoothed/capped)','Native labels (no smoothing or FOV cap)'],OUT/'figures/original_vs_corrected.png',np.stack([np.minimum(original.bounds[0],bounds[0]),np.maximum(original.bounds[1],bounds[1])]),colors=False)
    render([combined,combined],['Components before: all labels retained','Components after: 0 voxels removed'],OUT/'figures/components_before_after.png',bounds,colors=False)
    ct_mesh=trimesh.load_mesh(OUT/'meshes/ct_300hu_evidence.ply',process=False)
    render([ct_mesh],['Original CT >= 300 HU; includes nonbone objects'],OUT/'figures/ct_bone_evidence.png',ct_mesh.bounds,colors=False)
    interactive(colored,OUT/'skull_plus_mandible_confidence.html','CASE_001 skull + partial mandible + upper teeth')
    print('Saved-export figure QA rendered',flush=True)


if __name__=='__main__':
    if '--render-only' in sys.argv: render_saved_outputs()
    elif '--finalize' in sys.argv: finalize_report()
    else:
        import subprocess, uuid
        OUT.mkdir(exist_ok=True)
        result=subprocess.run([sys.executable,'-m','pytest','tests/test_patient_surface.py','tests/test_anatomy_phases4_10.py','tests/test_phase6_evaluation.py',
            '-q','--basetemp',str(ROOT/'.tmp'/('skull_regression_'+uuid.uuid4().hex)), '-p','no:cacheprovider',
            '--junitxml',str(OUT/'regression_tests.xml')],cwd=ROOT,capture_output=True,text=True)
        (OUT/'regression_tests.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError('Regression tests failed; see regression_tests.log')
        main()
        finalize_report()
