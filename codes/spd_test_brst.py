import pyspedas as spd
import pytplot as ptt
import matplotlib.pyplot as plt
import pyspedas.mms.cotrans.mms_cotrans_lmn as mms_cotrans_lmn

# trange = ['2015-09-02 16:38:44.145', '2015-09-02 16:58:44.145']
trange = ['2015-09-07 13:06:00', '2015-09-07 13:16:00']
data_rate = 'srvy'
probe = 3
time_clip = True

# _ = spd.mms.fgm(trange=trange, probe=probe, time_clip=time_clip, latest_version=True,
#                 get_fgm_ephemeris=True, varnames=mms_fgm_varnames, level='l2')

mms_fgm_varnames = [f'mms{probe}_fgm_b_gsm_srvy_l2', f'mms{probe}_fgm_b_gse_srvy_l2',
                    f'mms{probe}_fgm_b_gsm_lmn_srvy_l2_bvec',
                    f'mms{probe}_fgm_b_gse_lmn_srvy_l2_bvec']
_ = spd.mms.fgm(
     trange=trange,
     probe=probe,
     time_clip=time_clip,
     latest_version=True,
     get_fgm_ephemeris=True,
     varnames=mms_fgm_varnames,
     level="l2",
 )

fgm_time_unix = ptt.get_data('mms3_fgm_b_gsm_srvy_l2_bvec')[0]
fgm_b_gsm = ptt.get_data('mms3_fgm_b_gsm_srvy_l2_bvec')[1]
fgm_b_gse = ptt.get_data('mms3_fgm_b_gse_srvy_l2_bvec')[1]

_ = mms_cotrans_lmn.mms_cotrans_lmn(name_in=f'mms{probe}_fgm_b_gsm_srvy_l2_bvec',
                                    name_out=f'mms{probe}_fgm_b_gsm_lmn_srvy_l2',
                                    gsm=True, probe=str(probe), data_rate=data_rate)

_ = mms_cotrans_lmn.mms_cotrans_lmn(name_in=f'mms{probe}_fgm_b_gsm_srvy_l2_bvec',
                                    name_out=f'mms{probe}_fgm_b_gse_lmn_srvy_l2',
                                    gse=True, probe=str(probe), data_rate=data_rate)

fgm_b_gsm_lmn = ptt.get_data(f'mms{probe}_fgm_b_gsm_lmn_srvy_l2')[1]
fgm_b_gse_lmn = ptt.get_data(f'mms{probe}_fgm_b_gse_lmn_srvy_l2')[1]

fgm_time_utc = spd.time_datetime(fgm_time_unix)

# Plot all the data
fig, ax = plt.subplots(4, 1, sharex=True)
# Set figure szie
fig.set_size_inches(12, 8)
ax[0].plot(fgm_time_utc, fgm_b_gsm[:, 0], 'b', label="$B_{x}$")
ax[0].plot(fgm_time_utc, fgm_b_gsm[:, 1], 'g', label="$B_{y}$")
ax[0].plot(fgm_time_utc, fgm_b_gsm[:, 2], 'r', label="$B_{z}$")
ax[0].set_ylabel("B [nT] (GSM)")
ax[0].legend()
ax[0].set_title(f"MMS{probe} FGM data for {trange[0]} to {trange[1]}")

ax[1].plot(fgm_time_utc, fgm_b_gse[:, 0], 'b', label="$B_{x}$")
ax[1].plot(fgm_time_utc, fgm_b_gse[:, 1], 'g', label="$B_{y}$")
ax[1].plot(fgm_time_utc, fgm_b_gse[:, 2], 'r', label="$B_{z}$")
ax[1].set_ylabel("B [nT] (GSE)")
ax[1].legend()

ax[2].plot(fgm_time_utc, fgm_b_gsm_lmn[:, 0], 'b', label="$B_{l}$")
ax[2].plot(fgm_time_utc, fgm_b_gsm_lmn[:, 1], 'g', label="$B_{m}$")
ax[2].plot(fgm_time_utc, fgm_b_gsm_lmn[:, 2], 'r', label="$B_{n}$")
ax[2].set_ylabel("B [nT] (GSM, L-M-N)")
ax[2].legend()

ax[3].plot(fgm_time_utc, fgm_b_gse_lmn[:, 0], 'b', label="$B_{l}$")
ax[3].plot(fgm_time_utc, fgm_b_gse_lmn[:, 1], 'g', label="$B_{m}$")
ax[3].plot(fgm_time_utc, fgm_b_gse_lmn[:, 2], 'r', label="$B_{n}$")
ax[3].set_ylabel("B [nT] (GSE, L-M-N)")
ax[3].legend()

plt.savefig(f"../figures/fgm_data_01.png", dpi=300, bbox_inches="tight")
plt.close("all")

data_rate = 'fast'
data_type = ['des-moms', 'dis-moms']
mms_fpi_varnames = [f'mms{probe}_dis_bulkv_gse_{data_rate}',
                    f'mms{probe}_dis_bulkv_gsm_{data_rate}',
                    f'mms{probe}_dis_bulkv_gsm_lmn_{data_rate}',
                    f'mms{probe}_dis_bulkv_gse_lmn_{data_rate}']

_ = spd.mms.fpi(trange=trange, probe=probe, data_rate=data_rate, level='l2',
                datatype=data_type, time_clip=time_clip, varnames=mms_fpi_varnames,
                latest_version=True, get_support_data=True)


_ = spd.cotrans(name_in=f'mms{probe}_dis_bulkv_gse_{data_rate}',
                name_out=f'mms{probe}_dis_bulkv_gsm_{data_rate}', coord_in='gse',
                coord_out='gsm')

_ = mms_cotrans_lmn.mms_cotrans_lmn(name_in=f'mms{probe}_dis_bulkv_gsm_{data_rate}',
                                    name_out=f'mms{probe}_dis_bulkv_gsm_lmn_{data_rate}',
                                    gse=False, gsm=True, probe=str(probe), 
                                    data_rate=data_rate)

# _ = mms_cotrans_lmn.mms_cotrans_lmn(name_in=f'mms{probe}_dis_bulkv_gse_{data_rate}',
#                                     name_out=f'mms{probe}_dis_bulkv_gse_lmn_{data_rate}',
#                                     gse=True, gsm=False, probe=str(probe), data_rate=data_rate)

fpi_time_unix = ptt.get_data(mms_fpi_varnames[0])[0]
fpi_v_gse = ptt.get_data(mms_fpi_varnames[0])[1:][0]
fpi_v_gsm = ptt.get_data(mms_fpi_varnames[1])[1:][0]
fpi_v_gsm_lmn = ptt.get_data(mms_fpi_varnames[2])[1:][0]
#fpi_v_gse_lmn = ptt.get_data(mms_fpi_varnames[3])[1:][0]

fpi_time_utc = spd.time_datetime(fpi_time_unix)

# Print the first and last time in the data
print(f"\033[1;32m Start time: {fpi_time_utc[0].strftime('%Y-%m-%d %H:%M:%S')} \033[0m")
print(f"\033[1;32m End time: {fpi_time_utc[-1].strftime('%Y-%m-%d %H:%M:%S')} \033[0m")

# Plot all the data
fig, ax = plt.subplots(3, 1, sharex=True)
fig.set_size_inches(12, 8)
ax[0].plot(fpi_time_utc, fpi_v_gsm[:, 0], 'b', label="$V_{x}$")
ax[0].plot(fpi_time_utc, fpi_v_gsm[:, 1], 'g', label="$V_{y}$")
ax[0].plot(fpi_time_utc, fpi_v_gsm[:, 2], 'r', label="$V_{z}$")

ax[0].set_ylabel("V [km/s] (GSM)")
ax[0].legend()
ax[0].set_title(f"MMS{probe} FPI {data_rate} data")

ax[1].plot(fpi_time_utc, fpi_v_gse[:, 0], 'b', label="$V_{x}$")
ax[1].plot(fpi_time_utc, fpi_v_gse[:, 1], 'g', label="$V_{y}$")
ax[1].plot(fpi_time_utc, fpi_v_gse[:, 2], 'r', label="$V_{z}$")
ax[1].set_ylabel("V [km/s] (GSE)")
ax[1].legend()

ax[2].plot(fpi_time_utc, fpi_v_gsm_lmn[:, 0], 'b', label="$V_{l}$")
ax[2].plot(fpi_time_utc, fpi_v_gsm_lmn[:, 1], 'g', label="$V_{m}$")
ax[2].plot(fpi_time_utc, fpi_v_gsm_lmn[:, 2], 'r', label="$V_{n}$")
ax[2].set_ylabel("V [km/s] (GSM, L-M-N)")
ax[2].legend()

# plt.plot(fpi_time_utc, fpi_v_gse[:, 0], label='Vx_gse')
# plt.plot(fpi_time_utc, fpi_v_gse[:, 1], label='Vy_gse')
# plt.plot(fpi_time_utc, fpi_v_gse[:, 2], label='Vz_gse')
# plt.plot(fpi_time_utc, fpi_v_gsm[:, 0], label='Vx_gsm')
# plt.plot(fpi_time_utc, fpi_v_gsm[:, 1], label='Vy_gsm')
# plt.plot(fpi_time_utc, fpi_v_gsm[:, 2], label='Vz_gsm')
# plt.plot(fpi_time_utc, fpi_v_gse_lmn[:, 0], label='Vl_gse_lmn', lw=2, alpha=1, color='r')
# plt.plot(fpi_time_utc, fpi_v_gse_lmn[:, 1], label='Vm_gse_lmn', lw=2, alpha=1, color='b')
# plt.plot(fpi_time_utc, fpi_v_gse_lmn[:, 2], label='Vn_gse_lmn', lw=2, alpha=1, color='g')

# plt.plot(fpi_time_utc, fpi_v_gsm[:, 0], label='Vx_gsm', lw=1.5, alpha=0.3, color='b')
# plt.plot(fpi_time_utc, fpi_v_gsm[:, 1], label='Vy_gsm', lw=1.5, alpha=0.3, color='g')
# plt.plot(fpi_time_utc, fpi_v_gsm[:, 2], label='Vz_gsm', lw=1.5, alpha=0.3, color='r')
# 
# 
# plt.xlabel('Time [UTC]')
# plt.ylabel('V [km/s]')
# plt.xlim(fpi_time_utc[0], fpi_time_utc[-1])
# plt.legend()
# plt.title(f"MMS{probe} FPI {data_rate} data for "
#           f"{fpi_time_utc[0].strftime('%Y-%m-%d %H:%M:%S')} to "
#           f"{fpi_time_utc[-1].strftime('%Y-%m-%d %H:%M:%S')}")
plt.savefig(f"../figures/fpi_data.png", dpi=300, bbox_inches="tight")

