import numpy as np
import xarray as xr
from stk_xarray_conversion import stk_to_xarray, xarray_to_stk
from magnetic_field_lookup import magnetic_field
from rotation_methods import sheppards_method
import pysofa
import os.path
import datetime
from tqdm import tqdm


path = 'data\\CubeSatFixed'
if os.path.isfile(f'{path}.nc'):
    ephemeris_fixed = xr.open_dataset(f'{path}.nc')
else:
    ephemeris_fixed = stk_to_xarray(f'{path}.e')
    ephemeris_fixed.to_netcdf(f'{path}.nc')

path = 'data\\CubeSatFixed_NadirECF'
if os.path.isfile(f'{path}.nc'):
    attitude_fixed = xr.open_dataset(f'{path}.nc')
else:
    attitude_fixed = stk_to_xarray(f'{path}.a')
    attitude_fixed.to_netcdf(f'{path}.nc')

if ephemeris_fixed.attrs['ScenarioEpoch'] != attitude_fixed.attrs['ScenarioEpoch']:
    print('Ephemeris and attitude must have the same epoch')
    exit(1)
epoch = datetime.datetime.strptime(ephemeris_fixed.attrs['ScenarioEpoch'], '%d %b %Y %H:%M:%S.%f')
epoch_day = datetime.date(epoch.year, epoch.month, epoch.day)

n = ephemeris_fixed.dims['time']
times = ephemeris_fixed.time.values[:n]

q_nadir = np.zeros((n, 4))
q_rotate = np.zeros((n, 4))

omega = np.pi / 180.  # one deg/sec rotation

for i in tqdm(range(len(times))):
    loct = ephemeris_fixed.positions.isel(time=i).values
    bv = magnetic_field(epoch_day, loct)

    # get the local up direction
    lon, lat, height = pysofa.gc2gd(1, loct)
    loctplus1 = pysofa.gd2gc(1, lon, lat, height + 1.0)
    localup = loctplus1 - loct

    # choose the body frame (x -> mag field, z in plane with mag field and local down)
    bodyx = bv * (1.0 / np.linalg.norm(bv))
    bodyy = np.cross(-localup, bodyx).squeeze()
    bodyy *= 1.0 / np.linalg.norm(bodyy)
    bodyz = np.cross(bodyx, bodyy)
    bodyz *= 1.0 / np.linalg.norm(bodyz)

    # construct dcm, convert to quaternions
    bodydcm = np.stack((bodyx, bodyy, bodyz))
    q_nadir[i, :] = sheppards_method(bodydcm, scalar_last=True)

    # rotate around magnetic field vector
    costheta, sintheta = np.cos(omega * times[i]), np.sin(omega * times[i])
    bodyy, bodyz = costheta*bodyy + sintheta*bodyz, costheta*bodyz - sintheta*bodyy
    bodydcm = np.stack((bodyx, bodyy, bodyz))
    q_rotate[i, :] = sheppards_method(bodydcm, scalar_last=True)

attrs = attitude_fixed.attrs.copy()
attrs.pop('WrittenBy')
attrs['NumberOfAttitudePoints'] = n

data_vars = {'quaternions': (['time', 'q'], q_nadir)}
coords = {'time': ('time', times)}
ds_nadir = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds_nadir.to_netcdf('simulations\\magnetic_nadir.nc')
xarray_to_stk(ds_nadir, 'simulations\\magnetic_nadir.a')

data_vars = {'quaternions': (['time', 'q'], q_rotate)}
ds_rotate = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds_rotate.to_netcdf('simulations\\magnetic_rotate.nc')
xarray_to_stk(ds_rotate, 'simulations\\magnetic_rotate.a')

print(ds_nadir)
print(ds_rotate)