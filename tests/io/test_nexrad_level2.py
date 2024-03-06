#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar.io.nexrad_archive` module."""

import xarray as xr

from xradar.io.backends.nexrad_level2 import (
    NexradLevel2BackendEntrypoint,
    NEXRADLevel2File,
    open_nexradlevel2_datatree,
)


def test_open_nexradlevel2_datatree(nexradlevel2_file):
    dtree = open_nexradlevel2_datatree(nexradlevel2_file)
    print(dtree["/"])
    ds = dtree["sweep_0"]
    # assert ds.attrs["instrument_name"] == "KATX"
    # assert ds.attrs["nsweeps"] == 16
    # assert ds.attrs["Conventions"] == "CF/Radial instrument_parameters"
    assert ds["DBZH"].shape == (720, 1832)
    assert ds["DBZH"].dims == ("azimuth", "range")
    assert int(ds.sweep_number.values) == 0
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(ds["DBZH"].values)
    plt.figure()
    plt.imshow(ds["RHOHV"].values)
    plt.figure()
    plt.imshow(ds["ZDR"].values)
    plt.figure()
    plt.imshow(ds["PHIDP"].values)
    plt.show()


def test_open_nexrad_level2_backend(nexradlevel2_file):
    for i in range(16):
        ds = xr.open_dataset(
            nexradlevel2_file, engine=NexradLevel2BackendEntrypoint, group=f"sweep_{i}"
        )
        print(ds)
        # assert ds.attrs["instrument_name"] == "KATX"
        # assert ds.attrs["nsweeps"] == 16
        # assert ds.attrs["Conventions"] == "CF/Radial instrument_parameters"
        # assert ds["DBZH"].shape == (720, 1832)
        # assert ds["DBZH"].dims == ("azimuth", "range")
        # assert int(ds.sweep_number.values) == 0


def test_open_nexrad_level2_file_ds(nexradlevel2_file):
    ds = xr.open_dataset(nexradlevel2_file, engine="nexradlevel2", group="sweep_15")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    ds = ds.set_coords("sweep_mode")
    ds = ds.wrl.georef.georeference()

    print(ds.DBZH.attrs)
    print(ds.DBZH.encoding)
    print(ds.PHIDP.encoding)
    print(ds.PHIDP.values)
    ds.DBZH.wrl.vis.plot(ax=221)
    ds.ZDR.wrl.vis.plot(ax=222)
    ds.RHOHV.wrl.vis.plot(ax=223)
    ds.PHIDP.wrl.vis.plot(ax=224)
    plt.show()

    # import matplotlib.pyplot as plt

    # ref = fh.get_data("REF", 1832, scans=[0], raw_data=True)
    # plt.imshow(ref)
    # plt.show()


def test_open_nexrad_level2_file(nexradlevel2_file):
    print(nexradlevel2_file)
    fh = NEXRADLevel2File(nexradlevel2_file)
    print("--- Data Header ---")
    print(len(fh.data_header))
    # print(fh._data_headers[2000:3000])

    print("--- Meta Header ---")
    print(len(fh.meta_header))
    print(sum([len(x) for x in fh.meta_header]))

    print(fh._msg_5_data)

    print("--- MSG 31 Header ---")
    print("elevs", len(fh.msg_31_header))
    for i, el in enumerate(fh.msg_31_header):
        print(f"Sweep {i}")
        print("nrays", len(el), el[-1]["record_number"] - el[0]["record_number"])
        print("start -->", el[0])
        print("start+1 >", el[1])
        print("stop  -->", el[-1])

    print("--- MSG 31 Data Header ---")
    print("msg31 data", len(fh.msg_31_data_header))
    for i, dh in enumerate(fh.msg_31_data_header):
        print(i, dh)
    #
    # # print("--- Data Header ---")
    # # for i in range(len(scans_idx) - 1):
    # #     start = scans_idx[i]
    # #     stop = scans_idx[i+1] - 1
    # #     print(f"Sweep Data {i}")
    # #     print("start -->", start, fh._msg_31_data_headers[start])
    # #     print("stop -->", stop, fh._msg_31_data_headers[stop])
    #
    # # for i, hd in enumerate(fh._msg_31_data_headers):
    # #     print("elevation:", i)
    # #     for name, bh in hd.items():
    # #         print("--->:", name, bh)
    #
    # # fh.get_moment(0, "REF")
    # #for i in range(nelev):
    # i = 0
    # fh.get_sweep(i)
    # fh.get_data(i)
    #
    #
    # print("--- Moments Data ---")
    # for swpnr, sweep in fh._data.items():
    #     print("sweep nr:", swpnr)
    #     for name, bh in sweep.items():
    #         if name in ["sweep_data", "sweep_constant_data"]:
    #             print("--->:", name)
    #             for mom, mh in bh.items():
    #                 print("----->:", mom, mh)
    #         else:
    #             print("--->:", name, bh)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # print(fh.data[0]["sweep_data"]["PHI"]["data"])
    # plt.imshow(fh.data[0]["sweep_data"]["REF"]["data"])
    # plt.figure()
    # plt.imshow(fh.data[0]["sweep_data"]["ZDR"]["data"])
    # plt.figure()
    # plt.imshow(fh.data[0]["sweep_data"]["PHI"]["data"])
    # plt.figure()
    # plt.imshow(fh.data[0]["sweep_data"]["RHO"]["data"])
    # plt.show()

    # print(fh._data[0]["sweep_data"]["REF"])

    # for sidx in scans_idx:
    #     msg = fh._msg_31_headers[sidx]
    #     block_pointers = [
    #         v for k, v in msg.items() if k.startswith("block_pointer") and v > 0
    #     ]
    #     print(block_pointers)
    #     print(msg)
    #     # for block_pointer in block_pointers[:msg["block_count"]]:
    #     #     fh.rh.pos = block_pointer + LEN_MSG_HEADER
    #     #     # print(block_pointer)
    #     #     # print(self.filepos)
    #     #
    #     #     dheader = _unpack_dictionary(self._rh.read(4, width=1), DATA_BLOCK_HEADER,
    #     #                                  self._rawdata, byte_order=">")
    #     #
    #     #     block = DATA_BLOCK_TYPE_IDENTIFIER[dheader["block_type"]][
    #     #         dheader["data_name"]]
    #     #     LEN_BLOCK = struct.calcsize(_get_fmt_string(block, byte_order=">"))
    #     #     block_header = _unpack_dictionary(
    #     #         self._rh.read(LEN_BLOCK, width=1),
    #     #         block,
    #     #         self._rawdata,
    #     #         byte_order=">",
    #     #     )
    #
    # nscans = len(scan_msgs)
    #
    # print("nscans:", nscans)

    # print(fh._msg_5_data["elevation_data"])
    # print([(rec["type"], rec["size"], rec["seg_num"]) for rec in fh.raw_product_bhdrs])

    # elev_nums = np.array(
    #     [m["elevation_number"] for m in fh.raw_product_bhdrs]
    # )
    # print(elev_nums)
    # msg_18 = xradar.io.backends.nexrad_level2_new.MSG_18
    # print(xradar.io.backends.nexrad_level2_new.LEN_MSG_18)
    # print(xradar.io.backends.iris._get_fmt_string(msg_18, byte_order=">"))
    # for k, v in msg_18.items():
    #    print(k, v)
    # import time
    # time.sleep(3)
    # assert 1 == 2
