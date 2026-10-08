"""Physical longitude, resampling, and integral invariance contracts."""

from datetime import datetime
import importlib

import numpy as np
import pytest
from astropy.io import fits

from coconut_tools.magnetogram.core.coordinates import build_theta_phi, spherical_pixel_areas
from coconut_tools.magnetogram.io import readers
from coconut_tools.magnetogram.io.writers import write_bc_file
from coconut_tools.magnetogram.processing import longitude
from coconut_tools.magnetogram.processing.flux_balance import correct_net_flux
from coconut_tools.magnetogram.processing.spherical_harmonics import project_and_reconstruct


PRODUCTS = ['GONG', 'GONG_mrzqs', 'GONG_mrbqs', 'GONG_mrbqj', 'GONG_mrmqs',
            'GONG_mrnqs', 'ADAPT', 'HMI_fdt', 'HMI_small', 'HMI_polfil',
            'HMI_SYNC', 'HMI_hourly', 'custom']


def write_physical_map(path, product, start=180.3, frame='CRLN-CAR'):
    """Write a labeled grid; returned coordinates independently label columns."""
    step, width = 30., 12
    hmi = product.startswith('HMI') and product != 'HMI_fdt'
    native = start + np.arange(width) * step
    data = np.tile(native, (6, 1))
    if product in ('ADAPT', 'HMI_fdt'):
        data = np.stack([data, data + 1000])
    hdu = fits.PrimaryHDU(data)
    for key, value in dict(CTYPE1=frame, CUNIT1='deg', CRPIX1=1.,
                           CRVAL1=-start if product in ('HMI_SYNC', 'HMI_hourly') else (native[-1] if hmi else start),
                           CDELT1=-step if hmi else step,
                           CTYPE2='HGLT-CAR', CUNIT2='deg', CRPIX2=1.,
                           CRVAL2=-75., CDELT2=30., LNGTYPE=0).items():
        hdu.header[key] = value
    hdu.writeto(path)
    return data[0] if data.ndim == 3 else data, native


@pytest.mark.parametrize('product', PRODUCTS)
@pytest.mark.parametrize('resize', [False, True])
def test_physical_reader_centers_and_field_are_resampled_together(tmp_path, monkeypatch, product, resize):
    from skimage.transform import resize as resample
    path = tmp_path / 'map.fits'
    raw, native = write_physical_map(path, product)
    monkeypatch.setattr(readers, 'RESIZED_MAGNETOGRAM_SHAPE', (12, 24))
    field, theta, phi = readers.read_magnetogram(str(path), product, carrington=True, resize=resize)
    order = np.argsort(native % 360)
    expected = raw[::-1, order]
    centers = native[order] % 360
    if resize:
        expected = resample(expected, (12, 24), preserve_range=True, mode='edge', clip=False, anti_aliasing=True)
        centers = centers[0] - 15 + (np.arange(24) + .5) * 15
        order = np.argsort(centers % 360)
        centers = centers[order] % 360
        expected = expected[:, order]
    np.testing.assert_allclose(field, expected, atol=1e-12)
    np.testing.assert_allclose(np.degrees(phi[0]), centers, atol=1e-12)
    assert np.all((phi >= 0) & (phi < 2*np.pi))
    assert spherical_pixel_areas(theta[:, 0], phi[0]).sum() == pytest.approx(4*np.pi, abs=2e-14)


@pytest.mark.parametrize('offset', [.3, .5, 0.])
@pytest.mark.parametrize('cut', [0, 17, 180])
def test_periodic_areas_flux_corrections_and_harmonics_are_invariant(offset, cut):
    phi = np.radians(np.arange(360) + offset)
    theta = (np.arange(18) + .5) * np.pi / 18
    Theta, Phi = build_theta_phi(theta, phi)
    field = .3 + np.cos(Theta) + 2*np.sin(Theta)*np.cos(Phi)
    area = spherical_pixel_areas(theta, phi)
    rolled = np.roll(field, cut, axis=1)
    rolled_phi = np.roll(phi, cut)
    normalized, lon = longitude.normalize_to_carrington(rolled, np.degrees(rolled_phi))
    rotated, stony = longitude.rotate_carrington_to_stonyhurst(normalized, lon, 20.3)
    for transformed, axis in [(rolled, rolled_phi), (normalized, np.radians(lon)), (rotated, np.radians(stony))]:
        weights = spherical_pixel_areas(theta, axis)
        assert weights.sum() == pytest.approx(4*np.pi, abs=2e-14)
        for fn in [lambda b:b, np.abs, lambda b:np.maximum(b,0), lambda b:np.minimum(b,0)]:
            np.testing.assert_allclose(np.sum(fn(transformed)*weights), np.sum(fn(field)*area), atol=5e-13, rtol=1e-13)
        for method in ('surface_mean', 'polarity_scaling'):
            corrected = correct_net_flux(transformed, theta, axis, method)
            baseline = correct_net_flux(field, theta, phi, method)
            np.testing.assert_allclose(np.sort(corrected, axis=1), np.sort(baseline, axis=1), atol=5e-13)
    result, coeff = project_and_reconstruct(field, Theta, Phi, 2)
    rt, rp = build_theta_phi(theta, np.radians(stony))
    stony_result, stony_coeff = project_and_reconstruct(rotated, rt, rp, 2)
    np.testing.assert_allclose(np.sort(result, axis=1), np.sort(stony_result, axis=1), atol=2e-13)
    orders = np.array([0, 1, 0, 1, 2])
    np.testing.assert_allclose(stony_coeff, coeff*np.exp(1j*orders*np.radians(20.3)), atol=2e-13)


def test_stonyhurst_residual_is_written_to_cartesian_boundary(tmp_path):
    br = np.arange(360.)[None, :]
    rotated, phi = longitude.rotate_carrington_to_stonyhurst(br, np.arange(360.), 20.3)
    assert phi[0] == pytest.approx(359.7)
    assert phi[1] == pytest.approx(.7)
    path = tmp_path / 'boundary.dat'
    write_bc_file(str(path), rotated, np.array([np.pi/2]), np.radians(phi), 1.)
    saved = np.loadtxt(path, skiprows=2)
    np.testing.assert_allclose(np.degrees(np.arctan2(saved[:,1], saved[:,0])) % 360, phi, atol=1e-12)
    np.testing.assert_array_equal(saved[:,3], rotated[0])


def test_custom_stonyhurst_requires_source_metadata_and_roundtrips(tmp_path, monkeypatch):
    path = tmp_path / 'stony.fits'
    write_physical_map(path, 'custom', frame='HGLN-CAR')
    with pytest.raises(ValueError, match='source observation date'):
        readers.read_magnetogram(str(path), carrington=True)
    with fits.open(path, mode='update') as hdus:
        hdus[0].header['CRLN_OBS'] = 20.3
    br, th, ph = readers.read_magnetogram(str(path), carrington=True)
    monkeypatch.setattr(longitude, 'compute_carrington_central_meridian', lambda date:20.3)
    out, _, phi, _ = longitude.apply_configured_longitude_rotation(
        br, None, str(path), 'custom', datetime(2026,1,1), False, True,
        effective_date=datetime(2026,1,1), Phi=ph)
    np.testing.assert_allclose(np.mod(out[0],360), np.degrees(phi[0]), atol=1e-12)


def test_custom_physical_centers_are_never_snapped_to_zero(tmp_path):
    path = tmp_path / 'tiny_offset.fits'
    write_physical_map(path, 'custom', start=1e-8)
    _, _, phi = readers.read_magnetogram(str(path), carrington=True)
    assert np.degrees(phi[0,0]) == pytest.approx(1e-8, abs=1e-15)


def test_custom_decreasing_axis_reorders_field_with_coordinates(tmp_path):
    path = tmp_path / 'decreasing.fits'
    raw, _ = write_physical_map(path, 'custom', start=180.3)
    with fits.open(path, mode='update') as hdus:
        hdus[0].header['CDELT1'] = -30.
    br, _, phi = readers.read_magnetogram(str(path), carrington=True)
    native = (180.3-np.arange(12)*30.)%360
    order = np.argsort(native)
    np.testing.assert_array_equal(br,raw[::-1,order])
    np.testing.assert_allclose(np.degrees(phi[0]),native[order],atol=1e-12)


@pytest.mark.parametrize('bad', [np.arange(360.)+.01, np.arange(180.)*2, np.arange(360.)[::-1]])
def test_temporal_axes_cannot_be_combined_by_column_when_different(bad):
    with pytest.raises(RuntimeError, match='physical Carrington'):
        longitude.validate_matching_longitude_axes([np.arange(360.),bad], ['a','b'])


@pytest.mark.parametrize('offset', [-.000024, .000024, .00009])
def test_temporal_small_offsets_accept_and_align_across_seam(offset):
    reference = np.arange(3600)*.1 + .000012
    shifted = (reference + offset) % 360
    order = np.argsort(shifted)
    rolls = longitude.validate_matching_longitude_axes([reference, shifted[order]], ['a', 'b'])
    np.testing.assert_array_equal(np.roll(order, rolls[1]), np.arange(3600))


@pytest.mark.parametrize('product', ['HMI_hourly', 'HMI_SYNC'])
def test_dynamic_hmi_carrington_time_labels_each_native_column(tmp_path, product):
    path = tmp_path / 'dynamic.fits'
    raw = np.tile(np.arange(3600), (6, 1))
    hdu = fits.PrimaryHDU(raw)
    hdu.header.update(dict(CTYPE1='CRLN-CEA', CUNIT1='deg', CRPIX1=1800.,
        CRVAL1=832948.800012, CDELT1=-.1, CTYPE2='HGLT-CAR',
        CUNIT2='deg', CRPIX2=1., CRVAL2=-75., CDELT2=30.,
        LON_LAST=833128.700012, LON_FRST=832768.800012))
    hdu.writeto(path)
    br, _, phi = readers.read_magnetogram(str(path), product, carrington=True)
    expected = (-hdu.header['LON_LAST'] + np.arange(3600)*.1) % 360
    order = np.argsort(expected)
    np.testing.assert_array_equal(br, raw[::-1, order])
    np.testing.assert_allclose(np.degrees(phi[0]), expected[order], atol=2e-10, rtol=0)


@pytest.mark.parametrize('module_name', ['sph_filtering', 'gaussian_smoothing', 'NLD_implicit_method', 'Yaroslavsky_filter'])
@pytest.mark.parametrize('resize', [False, True])
def test_all_filters_stonyhurst_ignores_carrington_flag(tmp_path, monkeypatch, module_name, resize):
    module = importlib.import_module('coconut_tools.magnetogram.'+module_name)
    path = tmp_path / 'map.fits'
    write_physical_map(path, 'custom')
    monkeypatch.setattr(readers, 'RESIZED_MAGNETOGRAM_SHAPE', (12,24))
    monkeypatch.setattr(longitude, 'compute_carrington_central_meridian', lambda date:20.3)
    captured = []
    def capture(name, br, theta, phi, radius):
        captured.append((br.copy(), phi.copy()))
    monkeypatch.setattr(module, 'write_bc_file', capture)
    if module_name == 'sph_filtering':
        monkeypatch.setattr(module, 'project_and_reconstruct', lambda br,*args:(br,None))
    else:
        fn = 'filter_radial_field_weighted' if module_name == 'Yaroslavsky_filter' else 'filter_radial_field'
        monkeypatch.setattr(module, fn, lambda br,*args,**kwargs:(br, None) if module_name == 'NLD_implicit_method' else br)
    for flag in (False, True):
        module.process_magnetogram_date(dict(custom_magnetogram=str(path), carrington=flag,
            rotate_to_stonyhurst=True, resize=resize, show_map=False, write_map=True,
            output_dir=str(tmp_path), interpolation=False), datetime(2026,1,1))
    for a,b in zip(*captured):
        np.testing.assert_array_equal(a,b)
