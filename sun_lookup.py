import pysofa
import numpy as np
import icrf_to_fixed
import datetime


def sun(epoch: datetime.datetime, frame: str='ICRF'):
    """
    Returns a vector pointing at the sun at the time given by epoch in the requested frame.

    Currently supported frames: 'ICRF' and 'Fixed'.
    """
    djmjd0, date = pysofa.cal2jd(epoch.year, epoch.month, epoch.day)

    time = (60. * (60. * epoch.hour + epoch.minute) + epoch.second) / 86400.
    dat = pysofa.dat(epoch.year, epoch.month, epoch.day, time)
    utc = date + time
    tai = utc + dat / 86400.
    tt = tai + 32.184 / 86400.

    # pysofa.epv00 takes barycentric dynamical time (TDB), but terrestrial time (TT) is close enough
    h, b = pysofa.epv00(djmjd0, tt)
    hp = np.asarray(h[0, :]).squeeze()
    sun = -hp / np.linalg.norm(hp)

    if frame.lower() == 'icrf':
        return sun
    elif frame.lower() == 'fixed':
        R = icrf_to_fixed.icrf_to_fixed(epoch)
        return R @ sun
    else:
        raise ValueError(f'Unknown frame {frame}')


if __name__ == '__main__':
    epoch = datetime.datetime(year=2019, month=1, day=1, hour=12)
    print(sun(epoch))
