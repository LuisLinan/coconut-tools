"""Opt-in real-FITS longitude validation; run as a script, never during pytest.

Usage: python tests/validate_magnetogram_longitude.py --cache DIRECTORY
       [--hmi-sync FILE] [--baseline-readers FILE] [--download-stencils]
"""

import argparse
import importlib.util
import importlib
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from skimage.transform import resize as resize_image

from coconut_tools.magnetogram.core.coordinates import spherical_pixel_areas
from coconut_tools.magnetogram.io import readers
from coconut_tools.magnetogram.io.downloads import (
    download_interpolation_magnetograms, magnetogram_effective_date,
)
from coconut_tools.magnetogram.processing.longitude import apply_configured_longitude_rotation
from coconut_tools.tools.rotation_angle import compute_carrington_central_meridian


PATTERNS = dict(GONG='mrzqs*', GONG_mrzqs='mrzqs*', GONG_mrbqs='mrbqs*',
    GONG_mrbqj='mrbqj*', GONG_mrmqs='mrmqs*', GONG_mrnqs='mrnqs*',
    ADAPT='adapt403*', HMI_fdt='adapt40i*', HMI_small='hmi.Synoptic_Mr_small*',
    HMI_polfil='hmi.Synoptic_Mr_polfil*', HMI_hourly='hmi.synoptic_hourly*')


def independent_axis(header, product, n):
    """Decode provider metadata without calling the production axis helpers."""
    if product in ('HMI_hourly', 'HMI_SYNC'):
        axis = -(float(header['CRVAL1']) +
                 (np.arange(n)+1-float(header['CRPIX1']))*float(header['CDELT1']))
        np.testing.assert_allclose(axis[[0,-1]],
            [-header['LON_LAST'], -header['LON_FRST']], atol=1e-6, rtol=0)
        return axis
    if product.startswith('HMI') and product != 'HMI_fdt':
        # Evaluate in native index order, independently of production sorting.
        step = float(header['CDELT1'])
        first = float(header['CRVAL1']) + (n-float(header['CRPIX1']))*step
        axis = first + np.arange(n)*abs(step)
        # Redundant provider keywords have fewer/different printed decimals.
        np.testing.assert_allclose(axis[[0,-1]], [header['LON_FRST'],header['LON_LAST']], atol=1e-6, rtol=0)
        return axis
    return float(header['CRVAL1']) + (np.arange(n)+1-float(header['CRPIX1']))*float(header['CDELT1'])


def fluxes(br, theta, phi):
    weights = spherical_pixel_areas(theta, phi)
    return dict(area=float(weights.sum()), signed=float(np.sum(br*weights)),
        unsigned=float(np.sum(abs(br)*weights)), positive=float(np.sum(np.maximum(br,0)*weights)),
        negative=float(np.sum(np.minimum(br,0)*weights)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--hmi-sync', type=Path)
    parser.add_argument('--baseline-readers', type=Path)
    parser.add_argument('--download-stencils', action='store_true')
    parser.add_argument('--pipelines', action='store_true')
    args = parser.parse_args()
    baseline = None
    if args.baseline_readers:
        spec = importlib.util.spec_from_file_location('baseline_readers', args.baseline_readers)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    paths = {kind:next(iter(sorted(args.cache.glob(pattern))), None) for kind,pattern in PATTERNS.items()}
    paths['HMI_SYNC'] = args.hmi_sync
    paths['custom'] = paths['HMI_small']
    rows, diagnostics, stencils, pipelines, resize_comparisons = [], [], [], [], []
    for kind,path in paths.items():
        if path is None:
            rows.append(dict(type=kind,result='NOT VALIDATED',reason='No real file'))
            continue
        product = 'HMI_small' if kind == 'custom' else kind
        with fits.open(path) as hdus:
            hdu = next(h for h in hdus if h.data is not None and h.data.ndim>=2)
            header = hdu.header.copy()
            raw = np.asarray(hdu.data[0] if hdu.data.ndim==3 else hdu.data).copy()
        native = independent_axis(header, product, raw.shape[1])
        # All selected real products have south-to-north stored latitude rows.
        assert header['CDELT2'] > 0
        native = native % 360
        order = np.argsort(native)
        expected_native = raw[::-1, order]
        centers = native[order]
        date = magnetogram_effective_date(str(path), kind, '2026-09-01T00:00:00')
        l0 = compute_carrington_central_meridian(date)
        original_theta = readers.read_fits_theta_axis(str(path), product)[0]
        before = fluxes(np.nan_to_num(expected_native), original_theta, np.radians(centers))
        for resized in (False, True):
            legacy = readers.read_magnetogram(str(path), kind, resize=resized)
            legacy_ok = None if baseline is None else all(np.array_equal(a,b) for a,b in zip(
                legacy, baseline.read_magnetogram(str(path), kind, resize=resized)))
            rows.append(dict(type=kind,file=path.name,mode='legacy',resize=resized,interpolation=False,
                original_frame='Carrington',original_phi0=float(native[0]),final_frame='legacy',
                final_phi0=float(np.degrees(legacy[2][0,0])),max_error_deg=None,
                result='PASS' if legacy_ok else 'NOT VALIDATED'))
            expected = expected_native
            expected_centers = centers.copy()
            if resized:
                expected = resize_image(expected, readers.RESIZED_MAGNETOGRAM_SHAPE,
                    preserve_range=True, mode='edge', clip=False, anti_aliasing=True)
                new_n = expected.shape[1]
                expected_centers = (centers[0]-180/raw.shape[1]+(np.arange(new_n)+.5)*360/new_n) % 360
                order = np.argsort(expected_centers)
                expected_centers, expected = expected_centers[order], expected[:,order]
            br, th, ph = readers.read_magnetogram(str(path),kind,resize=resized,carrington=True)
            values_match = np.allclose(br,np.nan_to_num(expected),atol=1e-12,rtol=0)
            car_flux = fluxes(br,th[:,0],ph[0])
            if resized:
                alternative = resize_image(raw[::-1], readers.RESIZED_MAGNETOGRAM_SHAPE,
                    preserve_range=True, mode='edge', clip=False, anti_aliasing=True)
                new_step = 360/alternative.shape[1]
                alternative_centers = (native[0]-180/raw.shape[1]
                    +(np.arange(alternative.shape[1])+.5)*new_step)%360
                alternate_order = np.argsort(alternative_centers)
                alternative_centers = alternative_centers[alternate_order]
                alternative = np.nan_to_num(alternative[:,alternate_order])
                phase = float(abs((alternative_centers[0]-np.degrees(ph[0,0])
                    +new_step/2)%new_step-new_step/2))
                resize_comparisons.append(dict(type=kind,
                    normalize_first_phi0=float(np.degrees(ph[0,0])),
                    resize_first_phi0=float(alternative_centers[0]),
                    phase_difference_deg=phase,
                    max_field_difference_if_same_grid=float(np.max(abs(alternative-br)))
                    if phase < 1e-8 else None))
            for stony in (False,True):
                out,_,phi,angle = apply_configured_longitude_rotation(br,None,str(path),kind,date,False,stony,
                    effective_date=date,resize=resized,Phi=ph)
                reference_centers = expected_centers
                expected_out = expected
                if stony:
                    j = np.argmin(abs((reference_centers-l0+180)%360-180))
                    reference_centers = (np.roll(reference_centers,-j)-l0)%360
                    expected_out = np.roll(expected,-j,axis=1)
                error = float(np.max(abs((np.degrees(phi[0])-reference_centers+180)%360-180)))
                output_flux = fluxes(out,th[:,0],phi[0])
                invariant = all(np.isclose(output_flux[k],car_flux[k],atol=2e-9,rtol=1e-11) for k in car_flux)
                numerical_tolerance = 8*np.finfo(float).eps*max(360,abs(float(header['CRVAL1'])))
                ok = values_match and error < numerical_tolerance and invariant and np.allclose(out,np.nan_to_num(expected_out),atol=1e-12,rtol=0)
                rows.append(dict(type=kind,file=path.name,mode='Stonyhurst' if stony else 'Carrington',resize=resized,
                    interpolation=False,original_frame='Carrington',original_phi0=float(native[0]),
                    final_frame='Stonyhurst' if stony else 'Carrington',final_phi0=float(np.degrees(phi[0,0])),
                    max_error_deg=error,result='PASS' if ok else 'FAIL'))
                if stony:
                    diagnostics.append(dict(type=kind,resize=resized,before=before,carrington=car_flux,stonyhurst=output_flux,
                        resize_signed_delta=car_flux['signed']-before['signed'],
                        resize_unsigned_delta=car_flux['unsigned']-before['unsigned']))
        if kind.startswith('GONG'):
            historical_shift=int(path.name.split('_')[-1].split('.')[0])-1
            diagnostics.append(dict(type=kind,legacy_shift=historical_shift,
                legacy_first=float(np.roll(native,historical_shift)[0]),canonical_first=float(centers[0])))
        if args.pipelines:
            # A small global grid exercises the genuine numerical kernels and
            # Cartesian export affordably; the geometry above uses 360x720.
            for module_name in ('sph_filtering','gaussian_smoothing','NLD_implicit_method','Yaroslavsky_filter'):
                module=importlib.import_module('coconut_tools.magnetogram.'+module_name)
                output=args.cache/'pipeline_outputs'/f'{kind}_{module_name}.dat'
                try:
                    config=dict(map_type=kind,interpolation=False,resize=True,carrington=False,
                        rotate_to_stonyhurst=True,write_map=True,show_map=False,output_dir=str(output.parent),
                        adapt_map=0,lmax=2,iterations=1,template_size=3,apply_gaussian=False,Rn=1)
                    if kind=='custom':
                        config['custom_magnetogram']=str(path)
                    with patch.object(readers,'RESIZED_MAGNETOGRAM_SHAPE',(12,24)), patch.object(
                        module,'generate_output_and_map_names',return_value=(str(output),str(path))), patch(
                        'coconut_tools.magnetogram.filters.yaroslavsky.Pool',lambda:ThreadPool(2)):
                        result=module.process_magnetogram_date(config,date)
                    saved=np.loadtxt(result['output_name'],skiprows=2)
                    assert saved.shape==(288,4) and np.all(np.isfinite(saved))
                    np.testing.assert_allclose(np.linalg.norm(saved[:,:3],axis=1),1.,atol=1e-14)
                    pipelines.append(dict(type=kind,pipeline=module_name,result='PASS',file=result['output_name']))
                except Exception as exc:
                    pipelines.append(dict(type=kind,pipeline=module_name,result='FAIL',reason=repr(exc)))
    if args.download_stencils:
        for kind in ('GONG','GONG_mrbqs','GONG_mrbqj','ADAPT','HMI_fdt','HMI_hourly'):
            item = dict(type=kind)
            try:
                files,selection = download_interpolation_magnetograms('2026-09-01T00:00:00',kind,str(args.cache))
                item['headers'] = []
                for file in files:
                    with fits.open(file) as hs:
                        h = next(x.header for x in hs if x.data is not None and x.data.ndim>=2)
                        axis = np.sort(independent_axis(h,kind,h['NAXIS1'])%360)
                        if not item['headers']:
                            reference_axis = axis.copy()
                            axis_tolerance = 8*np.finfo(float).eps*max(360,abs(float(h['CRVAL1'])))
                        item['headers'].append(dict(file=Path(file).name,**{k:h.get(k) for k in ('NAXIS1','CRVAL1','CRPIX1','CDELT1')},first=float(axis[0]),last=float(axis[-1]),step=float(np.median(np.diff(axis)))))
                for resized in (False,True):
                    br,th,ph,linear=readers.read_interpolated_magnetogram(files,kind,selection,resize=resized,carrington=True)
                    for stony in (False,True):
                        out,_,phi,_=apply_configured_longitude_rotation(br,linear,files,kind,selection.target_date,True,stony,
                            effective_date=selection.target_date,Phi=ph)
                        a,b=fluxes(br,th[:,0],ph[0]),fluxes(out,th[:,0],phi[0])
                        assert all(np.isclose(a[k],b[k],atol=2e-9,rtol=1e-11) for k in a)
                        reference = reference_axis.copy()
                        if resized:
                            reference = np.sort((reference[0]-180/reference.size
                                +(np.arange(br.shape[1])+.5)*360/br.shape[1])%360)
                        if stony:
                            central = compute_carrington_central_meridian(selection.target_date)
                            column = np.argmin(abs((reference-central+180)%360-180))
                            reference = (np.roll(reference,-column)-central)%360
                        axis_error = float(np.max(abs((np.degrees(phi[0])-reference+180)%360-180)))
                        assert axis_error < axis_tolerance
                        rows.append(dict(type=kind,file='four-map stencil',mode='Stonyhurst' if stony else 'Carrington',
                            resize=resized,interpolation=True,original_frame='Carrington',
                            original_phi0=item['headers'][0]['first'],final_frame='Stonyhurst' if stony else 'Carrington',
                            final_phi0=float(np.degrees(phi[0,0])),max_error_deg=axis_error,result='PASS'))
                    if baseline:
                        old=baseline.read_interpolated_magnetogram(files,kind,selection,resize=resized)
                        new=readers.read_interpolated_magnetogram(files,kind,selection,resize=resized)
                        assert all(np.array_equal(a,b) for a,b in zip(old,new))
                item['result']='PASS'
            except Exception as exc:
                item['result']='REJECTED' if 'physical Carrington' in str(exc) else 'NOT VALIDATED'
                item['reason']=str(exc)
            stencils.append(item)
    report = dict(rows=rows,diagnostics=diagnostics,stencils=stencils,pipelines=pipelines,
        resize_order_comparison=resize_comparisons)
    target=args.cache/'validation.json'
    target.write_text(json.dumps(report,indent=2))
    print(target)
    print({status:sum(r['result']==status for r in rows) for status in ('PASS','FAIL','NOT VALIDATED')})
    print([(s['type'],s['result']) for s in stencils])
    if any(item['result']=='FAIL' for item in rows+pipelines):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
