"""Opt-in, offline audit of the same real hourly/SYNC FITS and old pipeline.

python tests/validate_hmi_dynamic_longitude.py --cache DIR --sync FILE
    --baseline DIR --output FILE
Baseline DIR contains the pre-refactor readers.py and longitude.py.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from sunpy.map.sources.sdo import HMISynopticMap

from coconut_tools.magnetogram.io import readers
from coconut_tools.magnetogram.io.downloads import (
    MagnetogramCandidate, select_interpolation_stencil, magnetogram_effective_date,
)
from coconut_tools.magnetogram.processing import longitude
from coconut_tools.tools.rotation_angle import compute_carrington_central_meridian


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    with fits.open(path) as hdus:
        hdu = next(h for h in hdus if h.data is not None)
        return hdu.data.copy(), hdu.header.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('cache', 'sync', 'baseline', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    old_readers = load(args.baseline/'readers.py', 'old_readers')
    old_longitude = load(args.baseline/'longitude.py', 'old_longitude')
    hourly = sorted(args.cache.glob('hmi.synoptic_hourly*.fits'))
    report = dict(files=[], feature_shifts=[], interpolation=[])
    for kind, path in [('HMI_hourly', p) for p in hourly] + [('HMI_SYNC', args.sync)]:
        raw, h = read(path)
        n = raw.shape[1]
        wcs = h['CRVAL1']+(np.arange(n)+1-h['CRPIX1'])*h['CDELT1']
        native = (-wcs) % 360
        endpoints = np.array([-h['LON_LAST'], -h['LON_FRST']]) % 360
        endpoint_error = float(np.max(abs(native[[0,-1]]-endpoints)))
        assert endpoint_error < 1e-6
        date = magnetogram_effective_date(str(path), kind, '2026-09-01T00:00:00')
        row = dict(product=kind, file=path.name,
            metadata={k:h.get(k) for k in ('TELESCOP','CONTENT','CTYPE1','CUNIT1','CUNIT2',
                'CRPIX1','CRVAL1','CDELT1','LON_FRST','LON_LAST')},
            sunpy_synoptic_match=bool(HMISynopticMap.is_datasource_for(raw,h)),
            reversed_wcs_first=float(wcs[-1]%360),
            abs_cdelt_first=float((h['CRVAL1']+(1-h['CRPIX1'])*abs(h['CDELT1']))%360),
            physical_first=float(native[0]), endpoint_error_deg=endpoint_error, comparisons=[])
        order = np.argsort(native)
        for resized in (False, True):
            br, _, phi = readers.read_magnetogram(str(path), kind, resize=resized, carrington=True)
            if not resized:
                np.testing.assert_array_equal(br, np.nan_to_num(raw[::-1, order]))
                np.testing.assert_allclose(np.degrees(phi[0]), native[order], atol=1e-9, rtol=0)
            new, _, _, _ = longitude.apply_configured_longitude_rotation(
                br, None, str(path), kind, date, False, True, effective_date=date,
                resize=resized, Phi=phi)
            old_br, _, _ = old_readers.read_magnetogram(str(path), kind, resize=resized)
            old, _, _ = old_longitude.apply_configured_longitude_rotation(
                old_br, None, str(path), kind, date, False, True, effective_date=date, resize=resized)
            same = bool(np.array_equal(new, old))
            if not resized:
                assert same, (kind, path.name, 'native Stonyhurst differs from baseline')
            row['comparisons'].append(dict(resize=resized, stonyhurst_equals_pre_refactor=same,
                max_abs_field_difference=float(np.max(abs(new-old)))))
            if resized:
                # Recover the actual centers sampled by the historical roll
                # and resize, independently of its artificial zero-based axis.
                step = 360 / old.shape[1]
                shift = int(np.round(wcs[0]/h['CDELT1'])) % n
                half_cell_change = (step-abs(h['CDELT1']))/2
                old_first = (native[-shift % n]+half_cell_change) % 360
                new_first = (native.min()+half_cell_change) % 360
                l0 = compute_carrington_central_meridian(date)
                axis = np.arange(old.shape[1])*step
                jold = np.argmin(abs((axis-l0+180)%360-180))
                jnew = np.argmin(abs((new_first+axis-l0+180)%360-180))
                row['comparisons'][-1]['physical_center_difference_from_pre_refactor_deg'] = float(
                    (new_first+jnew*step-old_first-jold*step+180)%360-180)
        report['files'].append(row)
    for before, after in zip(hourly, hourly[1:]):
        a, _ = read(before)
        b, _ = read(after)
        errors = {shift:float(np.nanmean((np.roll(a[200:1200], shift, axis=1)[:,1500:3300]
                  - b[200:1200,1500:3300])**2)) for shift in range(-12,13)}
        best = min(errors, key=errors.get)
        report['feature_shifts'].append(dict(before=before.name, after=after.name,
            best_shift_columns=best, mse=errors[best], neighbor_mse=[errors[best-1], errors[best+1]]))
    candidates = [MagnetogramCandidate(p.name, datetime.strptime(p.stem.split('_',2)[-1],
                  '%Y%m%d_%H%M%S'), None) for p in hourly]
    selection = select_interpolation_stencil(candidates, '2026-09-01T00:00:00')
    paths = list(map(str, hourly))
    for resized in (False, True):
        for physical in (False, True):
            br, _, phi, linear = readers.read_interpolated_magnetogram(
                paths, 'HMI_hourly', selection, resize=resized, carrington=physical)
            assert np.isfinite(br).all() and np.isfinite(linear).all()
            if physical:
                rotated, rotated_linear, _, _ = longitude.apply_configured_longitude_rotation(
                    br, linear, paths, 'HMI_hourly', selection.target_date, True, True, Phi=phi)
                np.testing.assert_array_equal(np.sort(rotated,axis=1), np.sort(br,axis=1))
                np.testing.assert_array_equal(np.sort(rotated_linear,axis=1), np.sort(linear,axis=1))
            report['interpolation'].append(dict(resize=resized, carrington=physical, result='PASS'))
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('PASS: real HMI columns, native baseline Stonyhurst, temporal interpolation; '+str(args.output))


if __name__ == '__main__':
    main()
