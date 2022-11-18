#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial2 output
================

This sub-module contains the writer for export of CfRadial2-based radar
data.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.to_cfradial2(dtree, filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "to_cfradial2",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from datatree import DataTree

from ...model import conform_cfradial2_sweep_group
from ...util import has_import
from ...version import version


def to_cfradial2(dtree, filename, timestep=None, engine=None):
    """Save DataTree to CfRadial2 compliant file.

    Parameters
    ----------
    dtree : DataTree
        DataTree with CfRadial2 groups.
    filename : str
        output filename

    Keyword Arguments
    ----------------
    timestep : int
        timestep of wanted volume, currently not used
    engine : str
        Either `netcdf4` or `h5netcdf`.
    """
    if engine is None:
        if has_import("netCDF4"):
            engine == "netcdf4"
        elif has_import("h5netcdf"):
            engine == "h5netcdf"
        else:
            raise ImportError(
                "wradlib: ``netCDF4`` or ``h5netcdf`` needed to perform this operation."
            )

    # iterate over DataTree and make subgroups cfradial2 compliant
    for grp in dtree.groups:
        if "sweep" in grp:
            dtree[grp] = DataTree(
                conform_cfradial2_sweep_group(
                    dtree[grp].to_dataset(), optional=False, dim0="azimuth"
                )
            )

    root = dtree["/"].to_dataset()
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    root.attrs["history"] += f": xradar v{version} CfRadial2 export"
    dtree["/"] = DataTree(root)

    dtree.to_netcdf(filename)
    # root.to_netcdf(filename, mode="w", group="/", engine=engine)
    # for idx, key in enumerate(root.sweep_group_name.values):
    #     ds = volume[idx]
    #     if "time" not in ds.dims:
    #         ds = ds.expand_dims("time")
    #     swp = ds.isel(time=timestep)
    #     swp.load()
    #     dim0 = list(set(swp.dims) & {"azimuth", "elevation"})[0]
    #     try:
    #         swp = swp.swap_dims({dim0: "time"})
    #     except ValueError:
    #         swp = swp.drop_vars("time").rename({"rtime": "time"})
    #         swp = swp.swap_dims({dim0: "time"})
    #     swp = swp.drop_vars(["x", "y", "z", "gr", "rays", "bins"], errors="ignore")
    #     swp = swp.sortby("time")
    #     swp.to_netcdf(filename, mode="a", group=key, engine=engine)
